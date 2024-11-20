use crate::loader::{ModelError, ModelLoader, XGBoostParser};
use crate::objective::Objective;
use crate::predicates::{AutoPredicate, Condition, Predicate};
use crate::tree::SplitType;

use super::binary_tree::{BinaryTree, BinaryTreeNode, DTNode};
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Float64Builder, Int64Array,
};
use arrow::datatypes::DataType;
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FeatureTreeError {
    #[error("Feature names must be provided")]
    MissingFeatureNames,
    #[error("Feature types must be provided")]
    MissingFeatureTypes,
    #[error("Feature names and types must have the same length")]
    LengthMismatch,
    #[error("Feature index {0} out of bounds")]
    InvalidFeatureIndex(usize),
    #[error("Invalid node structure")]
    InvalidStructure(String),
}

enum NodeDefinition {
    Leaf {
        weight: f64,
    },
    Split {
        feature_index: i32,
        split_value: f64,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTree {
    #[serde(with = "self::serde_helpers::binary_tree_serde")]
    pub(crate) tree: BinaryTree,
    pub(crate) feature_offset: usize,
    #[serde(with = "self::serde_helpers::arc_vec_serde")]
    pub(crate) feature_names: Arc<Vec<String>>,
    #[serde(with = "self::serde_helpers::arc_vec_serde")]
    pub(crate) feature_types: Arc<Vec<String>>,
}

mod serde_helpers {
    pub mod binary_tree_serde {
        use super::super::*;
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S>(tree: &BinaryTree, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serializer.serialize_newtype_struct("BinaryTree", &tree.nodes)
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<BinaryTree, D::Error>
        where
            D: Deserializer<'de>,
        {
            let nodes = Vec::deserialize(deserializer)?;
            Ok(BinaryTree { nodes })
        }
    }

    pub mod arc_vec_serde {
        use serde::{Deserialize, Deserializer, Serialize, Serializer};
        use std::sync::Arc;

        pub fn serialize<S, T>(arc: &Arc<Vec<T>>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
            T: Serialize,
        {
            Vec::serialize(arc, serializer)
        }

        pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Arc<Vec<T>>, D::Error>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de>,
        {
            let vec = Vec::deserialize(deserializer)?;
            Ok(Arc::new(vec))
        }
    }
}

impl fmt::Display for FeatureTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_node(
            f: &mut fmt::Formatter<'_>,
            tree: &FeatureTree,
            node: &BinaryTreeNode,
            prefix: &str,
            is_left: bool,
            feature_names: &[String],
        ) -> fmt::Result {
            let connector = if is_left { "├── " } else { "└── " };

            writeln!(
                f,
                "{}{}{}",
                prefix,
                connector,
                node_to_string(node, tree, feature_names)
            )?;

            if !node.value.is_leaf {
                let new_prefix = format!("{}{}   ", prefix, if is_left { "│" } else { " " });

                if let Some(left) = tree.tree.get_left_child(node) {
                    fmt_node(f, tree, left, &new_prefix, true, feature_names)?;
                }
                if let Some(right) = tree.tree.get_right_child(node) {
                    fmt_node(f, tree, right, &new_prefix, false, feature_names)?;
                }
            }
            Ok(())
        }

        fn node_to_string(
            node: &BinaryTreeNode,
            tree: &FeatureTree,
            feature_names: &[String],
        ) -> String {
            if node.value.is_leaf {
                format!("Leaf (weight: {:.4})", node.value.weight)
            } else {
                let feature_index = tree.feature_offset + node.value.feature_index as usize;
                let feature_name = feature_names
                    .get(feature_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.value.split_value)
            }
        }

        writeln!(f, "FeatureTree:")?;
        if let Some(root) = self.tree.get_node(self.tree.get_root_index()) {
            fmt_node(f, self, root, "", true, &self.feature_names)?;
        }
        Ok(())
    }
}

impl FeatureTree {
    pub fn new(feature_names: Arc<Vec<String>>, feature_types: Arc<Vec<String>>) -> Self {
        FeatureTree {
            tree: BinaryTree::new(),
            feature_offset: 0,
            feature_names,
            feature_types,
        }
    }

    #[inline(always)]
    pub fn predict(&self, features: &[f64]) -> f64 {
        let root = self
            .tree
            .get_node(self.tree.get_root_index())
            .expect("Tree should have root node");
        self.predict_one(root, features)
    }

    fn predict_one(&self, node: &BinaryTreeNode, features: &[f64]) -> f64 {
        if node.value.is_leaf {
            return node.value.weight;
        }

        let feature_idx = self.feature_offset + node.value.feature_index as usize;
        let split_value = unsafe { *features.get_unchecked(feature_idx) };

        if split_value < node.value.split_value {
            if let Some(left) = self.tree.get_left_child(node) {
                self.predict_one(left, features)
            } else {
                node.value.weight
            }
        } else if let Some(right) = self.tree.get_right_child(node) {
            self.predict_one(right, features)
        } else {
            node.value.weight
        }
    }

    #[inline(always)]
    pub fn predict_arrays(
        &self,
        feature_arrays: &[&dyn Array],
    ) -> Result<Float64Array, ArrowError> {
        let num_rows = feature_arrays[0].len();
        let mut builder = Float64Builder::with_capacity(num_rows);
        let mut row_features = vec![0.0; feature_arrays.len()];

        for row in 0..num_rows {
            for (i, array) in feature_arrays.iter().enumerate() {
                row_features[i] = array.as_primitive::<Float64Type>().value(row);
            }
            builder.append_value(self.predict(&row_features));
        }
        Ok(builder.finish())
    }

    pub fn depth(&self) -> usize {
        fn recursive_depth(tree: &BinaryTree, node: &BinaryTreeNode) -> usize {
            if node.value.is_leaf {
                1
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| recursive_depth(tree, n))
                    .unwrap_or(0)
                    .max(
                        tree.get_right_child(node)
                            .map(|n| recursive_depth(tree, n))
                            .unwrap_or(0),
                    )
            }
        }

        self.tree
            .get_node(self.tree.get_root_index())
            .map(|root| recursive_depth(&self.tree, root))
            .unwrap_or(0)
    }

    pub fn num_nodes(&self) -> usize {
        fn count_reachable_nodes(tree: &BinaryTree, node: &BinaryTreeNode) -> usize {
            if node.value.is_leaf {
                1
            } else {
                1 + tree
                    .get_left_child(node)
                    .map(|n| count_reachable_nodes(tree, n))
                    .unwrap_or(0)
                    + tree
                        .get_right_child(node)
                        .map(|n| count_reachable_nodes(tree, n))
                        .unwrap_or(0)
            }
        }

        self.tree
            .get_node(self.tree.get_root_index())
            .map(|root| count_reachable_nodes(&self.tree, root))
            .unwrap_or(0)
    }

    pub fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<FeatureTree> {
        if self.tree.is_empty() {
            return None;
        }

        let mut new_tree = FeatureTree::new(
            Arc::clone(&self.feature_names),
            Arc::clone(&self.feature_types),
        );
        new_tree.feature_offset = self.feature_offset;

        if let Some(root) = self.tree.get_node(self.tree.get_root_index()) {
            fn should_prune_direction(node: &DTNode, conditions: &[Condition]) -> Option<bool> {
                for condition in conditions {
                    match condition {
                        Condition::LessThan(value) => {
                            if *value <= node.split_value {
                                return Some(false); // Prune right path
                            }
                        }
                        Condition::GreaterThanOrEqual(value) => {
                            if *value >= node.split_value {
                                return Some(true); // Prune left path
                            }
                        }
                    }
                }
                None
            }
            #[allow(clippy::too_many_arguments)]
            fn prune_recursive(
                old_tree: &BinaryTree,
                new_tree: &mut BinaryTree,
                node: &BinaryTreeNode,
                feature_offset: usize,
                feature_names: &[String],
                predicate: &Predicate,
                parent_idx: Option<usize>,
                is_left: bool,
            ) -> Option<usize> {
                let new_node = node.value.clone();

                if !node.value.is_leaf {
                    let feature_index = feature_offset + node.value.feature_index as usize;
                    if let Some(feature_name) = feature_names.get(feature_index) {
                        if let Some(conditions) = predicate.conditions.get(feature_name) {
                            if let Some(prune_left) =
                                should_prune_direction(&node.value, conditions)
                            {
                                let child = if prune_left {
                                    old_tree.get_right_child(node)
                                } else {
                                    old_tree.get_left_child(node)
                                };

                                if let Some(child) = child {
                                    return prune_recursive(
                                        old_tree,
                                        new_tree,
                                        child,
                                        feature_offset,
                                        feature_names,
                                        predicate,
                                        parent_idx,
                                        is_left,
                                    );
                                }
                            }
                        }
                    }
                }

                let current_idx = if let Some(parent_idx) = parent_idx {
                    let new_tree_node = BinaryTreeNode::new(new_node);
                    if is_left {
                        new_tree.add_left_node(parent_idx, new_tree_node)
                    } else {
                        new_tree.add_right_node(parent_idx, new_tree_node)
                    }
                } else {
                    new_tree.add_root(BinaryTreeNode::new(new_node))
                };

                if !node.value.is_leaf {
                    if let Some(left) = old_tree.get_left_child(node) {
                        prune_recursive(
                            old_tree,
                            new_tree,
                            left,
                            feature_offset,
                            feature_names,
                            predicate,
                            Some(current_idx),
                            true,
                        );
                    }

                    if let Some(right) = old_tree.get_right_child(node) {
                        prune_recursive(
                            old_tree,
                            new_tree,
                            right,
                            feature_offset,
                            feature_names,
                            predicate,
                            Some(current_idx),
                            false,
                        );
                    }
                }

                Some(current_idx)
            }

            prune_recursive(
                &self.tree,
                &mut new_tree.tree,
                root,
                self.feature_offset,
                feature_names,
                predicate,
                None,
                true,
            );

            if !new_tree.tree.is_empty() {
                Some(new_tree)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn builder() -> FeatureTreeBuilder {
        FeatureTreeBuilder::new()
    }

    fn from_nodes(
        nodes: Vec<NodeDefinition>,
        feature_names: Arc<Vec<String>>,
        feature_types: Arc<Vec<String>>,
        feature_offset: usize,
    ) -> Result<Self, FeatureTreeError> {
        if nodes.is_empty() {
            return Err(FeatureTreeError::InvalidStructure("Empty tree".to_string()));
        }

        let mut binary_tree = BinaryTree::new();
        let mut node_map: HashMap<usize, usize> = HashMap::new();

        for (builder_idx, node_def) in nodes.iter().enumerate() {
            let tree_idx = match node_def {
                NodeDefinition::Split {
                    feature_index,
                    split_value,
                    ..
                } => {
                    let node = DTNode {
                        feature_index: *feature_index,
                        split_value: *split_value,
                        weight: 0.0,
                        is_leaf: false,
                        split_type: SplitType::Numerical,
                    };

                    if builder_idx == 0 {
                        binary_tree.add_root(node.into())
                    } else {
                        binary_tree.add_orphan_node(node.into())
                    }
                }

                NodeDefinition::Leaf { weight } => {
                    let node = DTNode {
                        feature_index: -1,
                        split_value: 0.0,
                        weight: *weight,
                        is_leaf: true,
                        split_type: SplitType::Numerical,
                    };

                    if builder_idx == 0 {
                        binary_tree.add_root(node.into())
                    } else {
                        binary_tree.add_orphan_node(node.into())
                    }
                }
            };
            node_map.insert(builder_idx, tree_idx);
        }

        for (builder_idx, node_def) in nodes.iter().enumerate() {
            if let NodeDefinition::Split { left, right, .. } = node_def {
                let parent_idx = node_map[&builder_idx];
                let left_idx = node_map[left];
                let right_idx = node_map[right];

                binary_tree
                    .connect_left(parent_idx, left_idx)
                    .map_err(|_| {
                        FeatureTreeError::InvalidStructure(
                            "Invalid left child connection".to_string(),
                        )
                    })?;
                binary_tree
                    .connect_right(parent_idx, right_idx)
                    .map_err(|_| {
                        FeatureTreeError::InvalidStructure(
                            "Invalid right child connection".to_string(),
                        )
                    })?;
            }
        }

        if !binary_tree.validate_connections() {
            return Err(FeatureTreeError::InvalidStructure(
                "Tree has disconnected nodes".into(),
            ));
        }

        Ok(Self {
            tree: binary_tree,
            feature_names,
            feature_types,
            feature_offset,
        })
    }
}

impl Default for FeatureTree {
    fn default() -> Self {
        FeatureTree::new(Arc::new(vec![]), Arc::new(vec![]))
    }
}

pub struct FeatureTreeBuilder {
    feature_names: Option<Arc<Vec<String>>>,
    feature_types: Option<Arc<Vec<String>>>,
    feature_offset: usize,
    split_indices: Vec<i32>,
    split_conditions: Vec<f64>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    base_weights: Vec<f64>,
}

impl FeatureTreeBuilder {
    pub fn new() -> Self {
        Self {
            feature_names: None,
            feature_types: None,
            feature_offset: 0,
            split_indices: Vec::new(),
            split_conditions: Vec::new(),
            left_children: Vec::new(),
            right_children: Vec::new(),
            base_weights: Vec::new(),
        }
    }

    pub fn feature_names(self, names: Vec<String>) -> Self {
        Self {
            feature_names: Some(Arc::new(names)),
            ..self
        }
    }

    pub fn feature_types(self, types: Vec<String>) -> Self {
        Self {
            feature_types: Some(Arc::new(types)),
            ..self
        }
    }

    pub fn feature_offset(self, offset: usize) -> Self {
        Self {
            feature_offset: offset,
            ..self
        }
    }

    pub fn split_indices(self, indices: Vec<i32>) -> Self {
        Self {
            split_indices: indices,
            ..self
        }
    }

    pub fn split_conditions(self, conditions: Vec<f64>) -> Self {
        Self {
            split_conditions: conditions,
            ..self
        }
    }

    pub fn children(self, left: Vec<u32>, right: Vec<u32>) -> Self {
        Self {
            left_children: left,
            right_children: right,
            ..self
        }
    }

    pub fn base_weights(self, weights: Vec<f64>) -> Self {
        Self {
            base_weights: weights,
            ..self
        }
    }

    pub fn build(self) -> Result<FeatureTree, FeatureTreeError> {
        let feature_names = self
            .feature_names
            .ok_or(FeatureTreeError::MissingFeatureNames)?;

        let feature_types = self
            .feature_types
            .ok_or(FeatureTreeError::MissingFeatureTypes)?;

        if feature_names.len() != feature_types.len() {
            return Err(FeatureTreeError::LengthMismatch);
        }

        let node_count = self.split_indices.len();
        if self.split_conditions.len() != node_count
            || self.left_children.len() != node_count
            || self.right_children.len() != node_count
            || self.base_weights.len() != node_count
        {
            return Err(FeatureTreeError::InvalidStructure(
                "Inconsistent array lengths in tree definition".to_string(),
            ));
        }

        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let is_leaf = self.left_children[i] == u32::MAX;
            let node = if is_leaf {
                NodeDefinition::Leaf {
                    weight: self.base_weights[i],
                }
            } else {
                NodeDefinition::Split {
                    feature_index: self.split_indices[i],
                    split_value: self.split_conditions[i],
                    left: self.left_children[i] as usize,
                    right: self.right_children[i] as usize,
                }
            };
            nodes.push(node);
        }

        FeatureTree::from_nodes(nodes, feature_names, feature_types, self.feature_offset)
    }
}

impl Default for FeatureTreeBuilder {
    fn default() -> Self {
        FeatureTreeBuilder::new()
    }
}

#[derive(Debug, Clone)]
pub struct GradientBoostedDecisionTrees {
    pub trees: Vec<FeatureTree>,
    pub feature_names: Arc<Vec<String>>,
    pub base_score: f64,
    pub feature_types: Arc<Vec<String>>,
    pub objective: Objective,
}

impl Default for GradientBoostedDecisionTrees {
    fn default() -> Self {
        GradientBoostedDecisionTrees {
            trees: vec![],
            feature_names: Arc::new(vec![]),
            feature_types: Arc::new(vec![]),
            base_score: 0.0,
            objective: Objective::SquaredError,
        }
    }
}

impl GradientBoostedDecisionTrees {
    pub fn predict_batch(&self, batch: &RecordBatch) -> Result<Float64Array, ArrowError> {
        self.predict_arrays(batch.columns())
    }

    pub fn predict_arrays(&self, feature_arrays: &[ArrayRef]) -> Result<Float64Array, ArrowError> {
        let num_rows = feature_arrays[0].len();
        let num_features = feature_arrays.len();
        let mut builder = Float64Builder::with_capacity(num_rows);

        let mut feature_values = Vec::with_capacity(num_features);
        feature_values.resize_with(num_features, Vec::new);

        for (i, array) in feature_arrays.iter().enumerate() {
            feature_values[i] = match array.data_type() {
                DataType::Float64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected Float64Array".into())
                        })?;
                    array.values().to_vec()
                }
                DataType::Int64 => {
                    let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Expected Int64Array".into())
                    })?;
                    array.values().iter().map(|&x| x as f64).collect()
                }
                DataType::Boolean => {
                    let array = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected BooleanArray".into())
                        })?;
                    array
                        .values()
                        .iter()
                        .map(|x| if x { 1.0 } else { 0.0 })
                        .collect()
                }
                _ => {
                    return Err(ArrowError::InvalidArgumentError(
                        "Unsupported feature type".into(),
                    ))?;
                }
            };
        }

        let mut row_features = vec![0.0; num_features];
        let num_trees = self.trees.len();

        if num_trees >= 100 {
            const BATCH_SIZE: usize = 8;
            let tree_batches = self.trees.chunks(BATCH_SIZE);
            let mut scores = vec![self.base_score; num_rows];

            for tree_batch in tree_batches {
                for row in 0..num_rows {
                    for (i, values) in feature_values.iter().enumerate() {
                        row_features[i] = values[row];
                    }

                    for tree in tree_batch {
                        scores[row] += tree.predict(&row_features);
                    }
                }
            }

            for score in scores {
                builder.append_value(self.objective.compute_score(score));
            }
        } else {
            for row in 0..num_rows {
                for (i, values) in feature_values.iter().enumerate() {
                    row_features[i] = values[row];
                }

                let mut score = self.base_score;
                for tree in &self.trees {
                    score += tree.predict(&row_features);
                }
                builder.append_value(self.objective.compute_score(score));
            }
        }

        Ok(builder.finish())
    }

    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn tree_depths(&self) -> Vec<usize> {
        self.trees.iter().map(|tree| tree.depth()).collect()
    }

    pub fn prune(&self, predicate: &Predicate) -> Self {
        let pruned_trees: Vec<FeatureTree> = self
            .trees
            .iter()
            .filter_map(|tree| tree.prune(predicate, &self.feature_names))
            .collect();

        GradientBoostedDecisionTrees {
            trees: pruned_trees,
            feature_names: self.feature_names.clone(),
            feature_types: self.feature_types.clone(),
            base_score: self.base_score,
            objective: self.objective.clone(),
        }
    }

    pub fn auto_prune(
        &self,
        batch: &RecordBatch,
        feature_names: &Arc<Vec<String>>,
    ) -> Result<Self, ArrowError> {
        let auto_predicate = AutoPredicate::new(Arc::clone(feature_names));
        let predicate = auto_predicate.generate_predicate(batch)?;
        Ok(self.prune(&predicate))
    }

    pub fn print_tree_info(&self) {
        println!("Total number of trees: {}", self.num_trees());

        let depths = self.tree_depths();
        println!("Tree depths: {:?}", depths);
        println!(
            "Average tree depth: {:.2}",
            depths.iter().sum::<usize>() as f64 / depths.len() as f64
        );
        println!("Max tree depth: {}", depths.iter().max().unwrap_or(&0));
        println!(
            "Total number of nodes: {}",
            self.trees
                .iter()
                .map(|tree| tree.num_nodes())
                .sum::<usize>()
        );
    }
}

impl ModelLoader for GradientBoostedDecisionTrees {
    fn load_from_json(json: &Value) -> Result<Self, ModelError> {
        let objective_type = XGBoostParser::parse_objective(json)?;

        let (feature_names, feature_types) = XGBoostParser::parse_feature_metadata(json)?;

        let base_score = json["learner"]["learner_model_param"]["base_score"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        let trees_json = json["learner"]["gradient_booster"]["model"]["trees"]
            .as_array()
            .ok_or_else(|| ModelError::MissingField("trees".to_string()))?;

        let trees = trees_json
            .iter()
            .map(|tree_json| {
                let arrays = XGBoostParser::parse_tree_arrays(tree_json)?;

                FeatureTreeBuilder::new()
                    .feature_names(feature_names.clone())
                    .feature_types(feature_types.clone())
                    .split_indices(arrays.split_indices)
                    .split_conditions(arrays.split_conditions)
                    .children(arrays.left_children, arrays.right_children)
                    .base_weights(arrays.base_weights)
                    .build()
                    .map_err(ModelError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            base_score,
            trees,
            feature_names: Arc::new(feature_names),
            feature_types: Arc::new(feature_types),
            objective: objective_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float64Array, Int64Array};
    use arrow::datatypes::Field;
    use arrow::datatypes::Schema;
    use std::sync::Arc;

    fn create_simple_tree() -> Result<FeatureTree, FeatureTreeError> {
        // Creates a simple decision tree:
        //          [age < 30]
        //         /          \
        //    [-1.0]        [income < 50k]
        //                  /           \
        //               [0.0]         [1.0]

        FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string(), "income".to_string()])
            .feature_types(vec!["numerical".to_string(), "numerical".to_string()])
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX],
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .build()
    }

    fn create_sample_tree() -> FeatureTree {
        // Create a simple tree:
        //          [feature0 < 0.5]
        //         /               \
        //    [-1.0]               [1.0]

        FeatureTreeBuilder::new()
            .feature_names(vec!["feature0".to_string()])
            .feature_types(vec!["numerical".to_string()])
            .split_indices(vec![0, -1, -1])
            .split_conditions(vec![0.5, 0.0, 0.0])
            .children(vec![1, u32::MAX, u32::MAX], vec![2, u32::MAX, u32::MAX])
            .base_weights(vec![0.0, -1.0, 1.0])
            .build()
            .unwrap()
    }

    fn create_sample_tree_deep() -> FeatureTree {
        // Create a deeper tree:
        //                    [feature0 < 0.5]
        //                   /               \
        //      [feature1 < 0.3]            [feature1 < 0.6]
        //     /               \            /               \
        // [feature2 < 0.7]    [-1.0]    [1.0]       [feature2 < 0.8]
        //   /        \                                /            \
        // [-2.0]    [2.0]                          [2.0]         [3.0]

        FeatureTreeBuilder::new()
            .feature_names(vec![
                "feature0".to_string(),
                "feature1".to_string(),
                "feature2".to_string(),
            ])
            .feature_types(vec![
                "float".to_string(),
                "float".to_string(),
                "float".to_string(),
            ])
            .split_indices(vec![0, 1, 2, -1, -1, 1, 2, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0])
            .children(
                vec![1, 3, 4, u32::MAX, u32::MAX, 7, 8, u32::MAX, u32::MAX],
                vec![5, 2, 6, u32::MAX, u32::MAX, 6, 8, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 2.0, 2.0, 3.0])
            .build()
            .unwrap()
    }

    fn create_sample_record_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("age", DataType::Float64, false),
            Field::new("income", DataType::Float64, false),
        ]);

        let age_array = Float64Array::from(vec![25.0, 35.0, 35.0, 28.0]);
        let income_array = Float64Array::from(vec![30000.0, 60000.0, 40000.0, 35000.0]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(age_array), Arc::new(income_array)],
        )
        .unwrap()
    }

    #[test]
    fn test_feature_tree_basic_predictions() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        // Test various prediction paths
        assert_eq!(tree.predict(&[25.0, 30000.0]), -1.0); // young age -> left path
        assert_eq!(tree.predict(&[35.0, 60000.0]), 1.0); // older age, high income -> right path
        assert_eq!(tree.predict(&[35.0, 40000.0]), 0.0); // older age, low income -> middle path

        Ok(())
    }

    #[test]
    fn test_feature_tree_array_predictions() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;
        let batch = create_sample_record_batch();

        let result = tree
            .predict_arrays(&[batch.column(0).as_ref(), batch.column(1).as_ref()])
            .unwrap();

        let expected = vec![-1.0, 1.0, 0.0, -1.0];
        assert_eq!(result.values(), expected.as_slice());

        Ok(())
    }

    #[test]
    fn test_feature_tree_serialization() -> Result<(), FeatureTreeError> {
        let original_tree = create_simple_tree()?;

        // Test serialization
        let serialized = serde_json::to_string(&original_tree).unwrap();
        let deserialized: FeatureTree = serde_json::from_str(&serialized).unwrap();

        // Test that predictions are the same after serialization
        let test_cases = vec![
            vec![25.0, 30000.0],
            vec![35.0, 60000.0],
            vec![35.0, 40000.0],
        ];

        for test_case in test_cases {
            assert_eq!(
                original_tree.predict(&test_case),
                deserialized.predict(&test_case),
                "Prediction mismatch for test case: {:?}",
                test_case
            );
        }

        Ok(())
    }

    #[test]
    fn test_feature_tree_builder_validation() {
        // Test missing feature names
        let result = FeatureTreeBuilder::new()
            .feature_types(vec!["numerical".to_string()])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![1], vec![2])
            .base_weights(vec![0.0])
            .build();
        assert!(matches!(result, Err(FeatureTreeError::MissingFeatureNames)));

        // Test length mismatch between feature names and types
        let result = FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string()])
            .feature_types(vec!["numerical".to_string(), "numerical".to_string()])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![1], vec![2])
            .base_weights(vec![0.0])
            .build();
        assert!(matches!(result, Err(FeatureTreeError::LengthMismatch)));
    }

    #[test]
    fn test_tree_depth_and_size() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        assert_eq!(tree.depth(), 3); // Root -> Income split -> Leaf
        assert_eq!(tree.num_nodes(), 5); // 2 internal nodes + 3 leaf nodes

        Ok(())
    }

    #[test]
    fn test_gbdt_basic() -> Result<(), FeatureTreeError> {
        let tree1 = create_simple_tree()?;
        let tree2 = create_simple_tree()?; // Using same tree structure for simplicity

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![tree1, tree2],
            feature_names: Arc::new(vec!["age".to_string(), "income".to_string()]),
            feature_types: Arc::new(vec!["numerical".to_string(), "numerical".to_string()]),
            base_score: 0.5,
            objective: Objective::SquaredError,
        };

        let batch = create_sample_record_batch();
        let predictions = gbdt.predict_batch(&batch).unwrap();

        assert_eq!(predictions.len(), 4);

        let expected_values: Vec<f64> = vec![-1.0, 1.0, 0.0, -1.0]
            .into_iter()
            .map(|x| 0.5 + 2.0 * x)
            .collect();

        for (i, &expected) in expected_values.iter().enumerate() {
            assert!((predictions.value(i) - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_different_array_types() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        // Test with different array types
        let age_array = Int64Array::from(vec![25, 35, 35, 28]);
        let income_array = BooleanArray::from(vec![false, true, false, false]);

        let result = tree.predict_arrays(&[&age_array, &income_array]).unwrap();

        // Verify predictions work with converted values
        assert_eq!(result.len(), 4);

        Ok(())
    }

    #[test]
    fn test_tree_predict_arrays() {
        let tree = create_sample_tree();

        let mut builder = Float64Builder::new();
        builder.append_value(0.3); // left: -1.0
        builder.append_value(0.7); // right: 1.0
        builder.append_value(0.5); // right: 1.0
        builder.append_value(0.0); // left: -1.0
        builder.append_null(); // default left: -1.0
        let array = Arc::new(builder.finish());
        let array_ref: &dyn Array = array.as_ref();

        let predictions = tree.predict_arrays(&[array_ref]).unwrap();

        assert_eq!(predictions.len(), 5);
        assert_eq!(predictions.value(0), -1.0); // 0.3 < 0.5 -> left
        assert_eq!(predictions.value(1), 1.0); // 0.7 >= 0.5 -> right
        assert_eq!(predictions.value(2), 1.0); // 0.5 >= 0.5 -> right
        assert_eq!(predictions.value(3), -1.0); // 0.0 < 0.5 -> left
        assert_eq!(predictions.value(4), -1.0); // default left (this needs to be fixed since
                                                // default values should come from default_true array in the model json)
    }

    #[test]
    fn test_pruning() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        // Create a predicate that forces age >= 30
        let mut conditions = HashMap::new();
        conditions.insert("age".to_string(), vec![Condition::GreaterThanOrEqual(30.0)]);
        let predicate = Predicate { conditions };

        // Prune the tree
        let pruned_tree = tree.prune(&predicate, &tree.feature_names).unwrap();

        // After pruning, all predictions should follow the right path
        let test_cases = vec![
            vec![25.0, 30000.0], // Would have gone left in original tree
            vec![35.0, 60000.0],
            vec![35.0, 40000.0],
        ];

        for test_case in test_cases {
            let prediction = pruned_tree.predict(&test_case);
            assert!(
                prediction == 0.0 || prediction == 1.0,
                "Prediction {} should only follow right path",
                prediction
            );
        }

        Ok(())
    }
    #[test]
    fn test_tree_prune() {
        let tree = create_sample_tree();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));
        let pruned_tree = tree.prune(&predicate, &["feature0".to_string()]).unwrap();
        assert_eq!(pruned_tree.tree.nodes.len(), 1);
        assert_eq!(pruned_tree.tree.get_node(0).unwrap().value.weight, -1.0);
    }

    #[test]
    fn test_tree_prune_deep() {
        let tree = create_sample_tree_deep();
        let feature_names = [
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        // Test case 1: Prune right subtree of root
        let mut predicate1 = Predicate::new();
        predicate1.add_condition("feature1".to_string(), Condition::LessThan(0.29));
        let pruned_tree1 = tree.prune(&predicate1, &feature_names).unwrap();
        assert_eq!(pruned_tree1.predict(&[0.6, 0.75, 0.8]), 2.0);

        // Test case 2: Prune left subtree of left child of root
        let mut predicate2 = Predicate::new();
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.70));
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();
        assert_eq!(pruned_tree2.predict(&[0.4, 0.6, 0.8]), -2.0);

        // Test case 3: Prune left root tree
        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        assert_eq!(pruned_tree3.predict(&[0.4, 0.6, 0.8]), 3.0);
    }

    #[test]
    fn test_tree_prune_multiple_conditions() {
        let tree = create_sample_tree_deep();
        let feature_names = vec![
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        // Test case 1: Multiple conditions affecting right path
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.4));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 2.0);
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 2.0);

        // Test case 2: Multiple conditions affecting left path
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), 3.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), -1.0);
    }

    #[test]
    fn test_xgboost_style_builder() -> Result<(), FeatureTreeError> {
        // This represents a simple tree:
        //          [age < 30]
        //         /          \
        //    [-1.0]        [income < 50k]
        //                  /           \
        //               [0.0]         [1.0]

        let tree = FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string(), "income".to_string()])
            .feature_types(vec!["numerical".to_string(), "numerical".to_string()])
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX], // left children
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX], // right children
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .build()?;

        // Test predictions
        assert!(tree.predict(&[25.0, 0.0]) < 0.0); // young
        assert!(tree.predict(&[35.0, 60000.0]) > 0.0); // old, high income
        assert!(tree.predict(&[35.0, 40000.0]) == 0.0); // old, low income

        Ok(())
    }

    #[test]
    fn test_array_length_mismatch() {
        let result = FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string()])
            .feature_types(vec!["numerical".to_string()])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![1], vec![2, 3]) // mismatched lengths
            .base_weights(vec![0.0])
            .build();

        assert!(matches!(result, Err(FeatureTreeError::InvalidStructure(_))));
    }
}
