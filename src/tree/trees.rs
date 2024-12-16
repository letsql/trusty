use super::vec_tree::{Traversable, TreeNode, VecTree};
use crate::loader::{ModelError, ModelLoader, XGBoostParser};
use crate::objective::Objective;
use crate::predicates::{Condition, Predicate};
use crate::tree::serde_helpers;
use crate::tree::{FeatureTreeError, FeatureType};
use arrow::array::{Array, ArrayRef, BooleanArray, Float32Array, Float32Builder, Int64Array};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

const CHUNK_SIZE: usize = 1024;
const TREE_BATCH_SIZE: usize = 16;

type VecTreeWithTreeNode = VecTree<TreeNode>;

#[derive(Debug)]
enum PruneAction {
    Keep,
    PruneLeft,
    PruneRight,
}

enum NodeDefinition {
    Leaf {
        weight: f32,
    },
    Split {
        feature_index: i32,
        default_left: bool,
        split_value: f32,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTree {
    #[serde(with = "self::serde_helpers::vec_tree_serde")]
    pub(crate) tree: VecTreeWithTreeNode,
    pub(crate) feature_offset: usize,
    #[serde(with = "self::serde_helpers::arc_vec_serde")]
    pub(crate) feature_names: Arc<Vec<String>>,
    #[serde(with = "self::serde_helpers::arc_vec_serde")]
    pub(crate) feature_types: Arc<Vec<FeatureType>>,
}

impl fmt::Display for FeatureTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_node(
            f: &mut fmt::Formatter<'_>,
            tree: &FeatureTree,
            node: &TreeNode,
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

            if !node.is_leaf() {
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

        fn node_to_string(node: &TreeNode, tree: &FeatureTree, feature_names: &[String]) -> String {
            if node.is_leaf() {
                format!("Leaf (weight: {:.4})", node.weight())
            } else {
                let feature_index = tree.feature_offset + node.feature_index() as usize;
                let feature_name = feature_names
                    .get(feature_index)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                format!("{} < {:.4}", feature_name, node.split_value())
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
    pub fn new(feature_names: Arc<Vec<String>>, feature_types: Arc<Vec<FeatureType>>) -> Self {
        FeatureTree {
            tree: VecTreeWithTreeNode::new(),
            feature_offset: 0,
            feature_names,
            feature_types,
        }
    }
    pub fn predict(&self, features: &[f32]) -> f32 {
        let mut current = match self.tree.get_node(self.tree.get_root_index()) {
            Some(node) => node,
            None => return 0.0,
        };

        loop {
            if current.is_leaf() {
                return current.weight();
            }

            let feature_idx = self.feature_offset + current.feature_index() as usize;
            let split_value = features[feature_idx];

            let go_right = if split_value.is_nan() {
                !current.default_left()
            } else {
                split_value >= current.split_value()
            };

            current = if go_right {
                match self.tree.get_right_child(current) {
                    Some(node) => node,
                    None => return current.weight(),
                }
            } else {
                match self.tree.get_left_child(current) {
                    Some(node) => node,
                    None => return current.weight(),
                }
            };
        }
    }

    pub fn depth(&self) -> usize {
        fn recursive_depth(tree: &VecTreeWithTreeNode, node: &TreeNode) -> usize {
            if node.is_leaf() {
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
        fn count_reachable_nodes(tree: &VecTreeWithTreeNode, node: &TreeNode) -> usize {
            if node.is_leaf() {
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
    #[inline]
    pub fn prune(&self, predicate: &Predicate, feature_names: &[String]) -> Option<FeatureTree> {
        if self.tree.is_empty() {
            return None;
        }

        let mut new_tree = FeatureTree::new(
            Arc::clone(&self.feature_names),
            Arc::clone(&self.feature_types),
        );
        new_tree.feature_offset = self.feature_offset;

        fn evaluate_node(
            node: &TreeNode,
            feature_index: usize,
            feature_names: &[String],
            predicate: &Predicate,
        ) -> PruneAction {
            if node.is_leaf() {
                return PruneAction::Keep;
            }

            if let Some(feature_name) = feature_names.get(feature_index) {
                if let Some(conditions) = predicate.conditions.get(feature_name) {
                    for condition in conditions {
                        match condition {
                            Condition::LessThan(value) => {
                                if node.should_prune_right(*value) {
                                    return PruneAction::PruneRight;
                                }
                            }
                            Condition::GreaterThanOrEqual(value) => {
                                if node.should_prune_left(*value) {
                                    return PruneAction::PruneLeft;
                                }
                            }
                        }
                    }
                }
            }
            PruneAction::Keep
        }

        fn prune_recursive(
            old_tree: &VecTreeWithTreeNode,
            new_tree: &mut VecTreeWithTreeNode,
            node_idx: usize,
            feature_offset: usize,
            feature_names: &[String],
            predicate: &Predicate,
        ) -> Option<usize> {
            let node = old_tree.get_node(node_idx)?;
            let feature_index = feature_offset + node.feature_index() as usize;

            match evaluate_node(node, feature_index, feature_names, predicate) {
                PruneAction::Keep => {
                    let new_idx = new_tree.nodes.len();
                    new_tree.nodes.push(node.clone());

                    if !node.is_leaf() {
                        let left_idx = prune_recursive(
                            old_tree,
                            new_tree,
                            node.left(),
                            feature_offset,
                            feature_names,
                            predicate,
                        );

                        let right_idx = prune_recursive(
                            old_tree,
                            new_tree,
                            node.right(),
                            feature_offset,
                            feature_names,
                            predicate,
                        );

                        if let Some(left_idx) = left_idx {
                            new_tree.connect_left(new_idx, left_idx).ok()?;
                        }
                        if let Some(right_idx) = right_idx {
                            new_tree.connect_right(new_idx, right_idx).ok()?;
                        }
                    }

                    Some(new_idx)
                }
                PruneAction::PruneLeft => prune_recursive(
                    old_tree,
                    new_tree,
                    node.right(),
                    feature_offset,
                    feature_names,
                    predicate,
                ),
                PruneAction::PruneRight => prune_recursive(
                    old_tree,
                    new_tree,
                    node.left(),
                    feature_offset,
                    feature_names,
                    predicate,
                ),
            }
        }

        let root_idx = self.tree.get_root_index();
        prune_recursive(
            &self.tree,
            &mut new_tree.tree,
            root_idx,
            self.feature_offset,
            feature_names,
            predicate,
        )?;

        Some(new_tree)
    }

    pub fn builder() -> FeatureTreeBuilder {
        FeatureTreeBuilder::new()
    }

    fn from_nodes(
        nodes: Vec<NodeDefinition>,
        feature_names: Arc<Vec<String>>,
        feature_types: Arc<Vec<FeatureType>>,
        feature_offset: usize,
    ) -> Result<Self, FeatureTreeError> {
        if nodes.is_empty() {
            return Err(FeatureTreeError::InvalidStructure("Empty tree".to_string()));
        }
        if feature_names.is_empty() {
            return Err(FeatureTreeError::MissingFeatureNames);
        }
        if feature_types.is_empty() {
            return Err(FeatureTreeError::MissingFeatureTypes);
        }

        if nodes.is_empty() {
            return Err(FeatureTreeError::InvalidStructure("Empty tree".to_string()));
        }

        let mut vec_tree = VecTreeWithTreeNode::new();
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        for (builder_idx, node_def) in nodes.iter().enumerate() {
            let tree_node = match node_def {
                NodeDefinition::Split {
                    feature_index,
                    split_value,
                    default_left,
                    ..
                } => TreeNode::new_split(*feature_index, *split_value, *default_left),
                NodeDefinition::Leaf { weight } => TreeNode::new_leaf(*weight),
            };

            let tree_idx = if builder_idx == 0 {
                vec_tree.add_root(tree_node)
            } else {
                vec_tree.add_orphan_node(tree_node)
            };

            node_map.insert(builder_idx, tree_idx);
        }

        for (builder_idx, node_def) in nodes.iter().enumerate() {
            if let NodeDefinition::Split { left, right, .. } = node_def {
                let parent_idx = node_map[&builder_idx];
                let left_idx = node_map[left];
                let right_idx = node_map[right];

                vec_tree.connect_left(parent_idx, left_idx).map_err(|_| {
                    FeatureTreeError::InvalidStructure("Invalid left child connection".to_string())
                })?;
                vec_tree.connect_right(parent_idx, right_idx).map_err(|_| {
                    FeatureTreeError::InvalidStructure("Invalid right child connection".to_string())
                })?;
            }
        }

        if !vec_tree.validate_connections() {
            return Err(FeatureTreeError::InvalidStructure(
                "Tree has disconnected nodes".into(),
            ));
        }

        Ok(Self {
            tree: vec_tree,
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
    feature_types: Option<Arc<Vec<FeatureType>>>,
    feature_offset: usize,
    split_indices: Vec<i32>,
    split_conditions: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    base_weights: Vec<f32>,
    default_left: Vec<bool>,
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
            default_left: Vec::new(),
        }
    }

    pub fn feature_names(self, names: Vec<String>) -> Self {
        Self {
            feature_names: Some(Arc::new(names)),
            ..self
        }
    }

    pub fn feature_types(self, types: Vec<FeatureType>) -> Self {
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

    pub fn split_conditions(self, conditions: Vec<f32>) -> Self {
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

    pub fn base_weights(self, weights: Vec<f32>) -> Self {
        Self {
            base_weights: weights,
            ..self
        }
    }

    pub fn default_left(self, indices: Vec<bool>) -> Self {
        Self {
            default_left: indices,
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
                    default_left: self.default_left[i],
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
    pub base_score: f32,
    pub feature_types: Arc<Vec<FeatureType>>,
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
    pub fn predict_batches(&self, batches: &[RecordBatch]) -> Result<Float32Array, ArrowError> {
        // Parallelize batch feature extraction and chunk creation
        let chunks: Vec<(Vec<Vec<f32>>, std::ops::Range<usize>)> = batches
            .par_iter()
            .map(|batch| -> Result<Vec<_>, ArrowError> {
                let feature_values = self.extract_features(batch.columns())?;
                let num_rows = feature_values[0].len();

                // Create chunks for this batch
                Ok((0..num_rows)
                    .step_by(CHUNK_SIZE)
                    .map(|start| {
                        let end = (start + CHUNK_SIZE).min(num_rows);
                        (feature_values.clone(), start..end)
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        self.predict_chunks(chunks)
    }

    #[inline]
    pub fn predict_arrays(&self, feature_arrays: &[ArrayRef]) -> Result<Float32Array, ArrowError> {
        // Extract features once
        let feature_values = self.extract_features(feature_arrays)?;
        let num_rows = feature_values[0].len();

        // Create chunks similar to predict_batches
        let chunks: Vec<(Vec<Vec<f32>>, std::ops::Range<usize>)> = (0..num_rows)
            .step_by(CHUNK_SIZE)
            .map(|start| {
                let end = (start + CHUNK_SIZE).min(num_rows);
                (feature_values.clone(), start..end)
            })
            .collect();

        self.predict_chunks(chunks)
    }

    // Common prediction logic for both methods
    fn predict_chunks(
        &self,
        chunks: Vec<(Vec<Vec<f32>>, std::ops::Range<usize>)>,
    ) -> Result<Float32Array, ArrowError> {
        // Process chunks in parallel
        let predictions: Vec<f32> = chunks
            .into_par_iter()
            .map(|(feature_values, range)| {
                let chunk_size = range.end - range.start;
                let mut feature_buffer = vec![0.0; feature_values.len()];
                let mut scores = vec![self.base_score; chunk_size];

                for tree_batch in self.trees.chunks(TREE_BATCH_SIZE) {
                    for (i, score) in scores.iter_mut().enumerate().take(chunk_size) {
                        let row_idx = range.start + i;

                        for (j, col) in feature_values.iter().enumerate() {
                            feature_buffer[j] = col[row_idx];
                        }

                        *score += tree_batch
                            .iter()
                            .map(|tree| tree.predict(&feature_buffer))
                            .sum::<f32>();
                    }
                }

                scores
                    .into_iter()
                    .map(|score| self.objective.compute_score(score))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        // Create the final array
        let mut builder = Float32Builder::with_capacity(predictions.len());
        builder.append_slice(&predictions);
        Ok(builder.finish())
    }

    #[inline]
    fn extract_features(&self, feature_arrays: &[ArrayRef]) -> Result<Vec<Vec<f32>>, ArrowError> {
        // Existing extract_features implementation remains the same
        let num_rows = feature_arrays[0].len();
        let mut feature_values = Vec::with_capacity(feature_arrays.len());

        for (array, feature_type) in feature_arrays.iter().zip(self.feature_types.iter()) {
            let values = match (array.data_type(), feature_type) {
                (DataType::Float32, FeatureType::Float) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected Float32Array".into())
                        })?;

                    let mut values = Vec::with_capacity(num_rows);
                    if let Some(null_bitmap) = array.nulls() {
                        let values_slice = array.values();
                        for i in 0..num_rows {
                            values.push(if null_bitmap.is_null(i) {
                                f32::NAN
                            } else {
                                values_slice[i]
                            });
                        }
                    } else {
                        values.extend_from_slice(array.values());
                    }
                    values
                }
                (DataType::Int64, FeatureType::Int) => {
                    let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ArrowError::InvalidArgumentError("Expected Int64Array".into())
                    })?;

                    let mut values = Vec::with_capacity(num_rows);
                    if let Some(null_bitmap) = array.nulls() {
                        let values_slice = array.values();
                        for i in 0..num_rows {
                            values.push(if null_bitmap.is_null(i) {
                                f32::NAN
                            } else {
                                values_slice[i] as f32
                            });
                        }
                    } else {
                        values.extend(array.values().iter().map(|&x| x as f32));
                    }
                    values
                }
                (DataType::Boolean, FeatureType::Indicator) => {
                    let array = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            ArrowError::InvalidArgumentError("Expected BooleanArray".into())
                        })?;

                    let mut values = Vec::with_capacity(num_rows);
                    if let Some(null_bitmap) = array.nulls() {
                        for i in 0..num_rows {
                            values.push(if null_bitmap.is_null(i) {
                                f32::NAN
                            } else if array.value(i) {
                                1.0
                            } else {
                                0.0
                            });
                        }
                    } else {
                        values.extend(array.values().iter().map(|x| if x { 1.0 } else { 0.0 }));
                    }
                    values
                }
                (actual, expected) => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Feature: expected {:?} for type {}, got {:?}",
                        expected.get_arrow_data_type(),
                        expected,
                        actual
                    )));
                }
            };

            feature_values.push(values);
        }

        Ok(feature_values)
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
        let base_score = XGBoostParser::parse_base_score(json)?;
        let trees_json = XGBoostParser::parse_trees(json)?;

        let trees = trees_json
            .iter()
            .map(|tree_json| {
                let arrays = XGBoostParser::parse_tree_arrays(tree_json)?;

                let tree = FeatureTreeBuilder::new()
                    .feature_names(feature_names.clone())
                    .feature_types(feature_types.clone())
                    .split_indices(arrays.split_indices)
                    .split_conditions(arrays.split_conditions)
                    .children(arrays.left_children, arrays.right_children)
                    .base_weights(arrays.base_weights)
                    .default_left(arrays.default_left)
                    .build()
                    .map_err(ModelError::from)?;
                Ok::<FeatureTree, ModelError>(tree)
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
    use arrow::array::Float32Array;
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
            .feature_types(vec![FeatureType::Float, FeatureType::Float])
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX],
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .default_left(vec![true, false, false, false, false])
            .build()
    }

    fn create_sample_tree() -> FeatureTree {
        // Create a simple tree:
        //          [feature0 < 0.5]
        //         /               \
        //    [-1.0]               [1.0]

        FeatureTreeBuilder::new()
            .feature_names(vec!["feature0".to_string()])
            .feature_types(vec![FeatureType::Float])
            .split_indices(vec![0, -1, -1])
            .split_conditions(vec![0.5, 0.0, 0.0])
            .children(vec![1, u32::MAX, u32::MAX], vec![2, u32::MAX, u32::MAX])
            .base_weights(vec![0.0, -1.0, 1.0])
            .default_left(vec![false, false, false])
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
                FeatureType::Float,
                FeatureType::Float,
                FeatureType::Float,
            ])
            .split_indices(vec![0, 1, 2, -1, -1, -1, 1, -1, 2, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.0, 0.6, 0.0, 0.8, 0.0, 0.0])
            .children(
                vec![
                    1,
                    3,
                    4,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    7,
                    u32::MAX,
                    9,
                    u32::MAX,
                    u32::MAX,
                ],
                vec![
                    6,
                    2,
                    5,
                    u32::MAX,
                    u32::MAX,
                    u32::MAX,
                    8,
                    u32::MAX,
                    10,
                    u32::MAX,
                    u32::MAX,
                ],
            )
            .base_weights(vec![
                0.0, 0.0, 0.0, -2.0, 2.0, -1.0, 0.0, 1.0, 0.0, 2.0, 3.0,
            ])
            .default_left(vec![
                true, true, true, false, false, true, false, false, true, false, false,
            ])
            .build()
            .unwrap()
    }

    fn create_sample_record_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("age", DataType::Float32, false),
            Field::new("income", DataType::Float32, false),
        ]);

        let age_array = Float32Array::from(vec![25.0, 35.0, 35.0, 28.0]);
        let income_array = Float32Array::from(vec![30000.0, 60000.0, 40000.0, 35000.0]);

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
    fn test_feature_tree_serialization() -> Result<(), FeatureTreeError> {
        let original_tree = create_simple_tree()?;

        let serialized = serde_json::to_string(&original_tree).unwrap();
        let deserialized: FeatureTree = serde_json::from_str(&serialized).unwrap();

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
            .feature_types(vec![FeatureType::Float])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![u32::MAX], vec![u32::MAX]) // Leaf node - no children
            .base_weights(vec![0.0])
            .default_left(vec![false])
            .build();
        assert!(matches!(result, Err(FeatureTreeError::MissingFeatureNames)));

        // Test length mismatch
        let result = FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string()])
            .feature_types(vec![FeatureType::Float, FeatureType::Float])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![u32::MAX], vec![u32::MAX]) // Leaf node - no children
            .base_weights(vec![0.0])
            .default_left(vec![false])
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
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.5,
            objective: Objective::SquaredError,
        };

        let batch = create_sample_record_batch();
        let predictions = gbdt.predict_batches(&[batch]).unwrap();

        assert_eq!(predictions.len(), 4);

        let expected_values: Vec<f32> = vec![-1.0, 1.0, 0.0, -1.0]
            .into_iter()
            .map(|x| 0.5 + 2.0 * x)
            .collect();

        for (i, &expected) in expected_values.iter().enumerate() {
            assert!((predictions.value(i) - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_pruning() -> Result<(), FeatureTreeError> {
        let tree = create_simple_tree()?;

        let mut conditions = HashMap::new();
        conditions.insert("age".to_string(), vec![Condition::GreaterThanOrEqual(30.0)]);
        let predicate = Predicate { conditions };

        let pruned_tree = tree.prune(&predicate, &tree.feature_names).unwrap();

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
        assert_eq!(pruned_tree.tree.get_node(0).unwrap().weight(), -1.0);
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
        predicate1.add_condition("feature1".to_string(), Condition::LessThan(0.30));
        let pruned_tree1 = tree.prune(&predicate1, &feature_names).unwrap();
        assert_eq!(pruned_tree1.predict(&[0.6, 0.75, 0.8]), 1.0);

        let mut predicate2 = Predicate::new();
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.70));
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();
        assert_eq!(pruned_tree2.predict(&[0.4, 0.2, 0.8]), -2.0);

        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        assert_eq!(pruned_tree3.predict(&[0.6, 0.7, 0.9]), 3.0);
    }

    #[test]
    fn test_tree_prune_multiple_conditions() {
        let tree = create_sample_tree_deep();
        let feature_names = vec![
            "feature0".to_string(),
            "feature1".to_string(),
            "feature2".to_string(),
        ];

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        predicate.add_condition("feature1".to_string(), Condition::LessThan(0.4));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.2, 0.0, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 1.0);

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));
        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), 1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), 1.0);
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
            .feature_types(vec![FeatureType::Float, FeatureType::Float])
            .split_indices(vec![0, -1, 1, -1, -1])
            .split_conditions(vec![30.0, 0.0, 50000.0, 0.0, 0.0])
            .children(
                vec![1, u32::MAX, 3, u32::MAX, u32::MAX], // left children
                vec![2, u32::MAX, 4, u32::MAX, u32::MAX], // right children
            )
            .base_weights(vec![0.0, -1.0, 0.0, 0.0, 1.0])
            .default_left(vec![false, false, false, false, false])
            .build()?;

        assert!(tree.predict(&[25.0, 0.0]) < 0.0); // young
        assert!(tree.predict(&[35.0, 60000.0]) > 0.0); // old, high income
        assert!(tree.predict(&[35.0, 40000.0]) == 0.0); // old, low income

        Ok(())
    }

    #[test]
    fn test_array_length_mismatch() {
        let result = FeatureTreeBuilder::new()
            .feature_names(vec!["age".to_string()])
            .feature_types(vec![FeatureType::Float])
            .split_indices(vec![0])
            .split_conditions(vec![30.0])
            .children(vec![1], vec![2, 3]) // mismatched lengths
            .base_weights(vec![0.0])
            .build();

        assert!(matches!(result, Err(FeatureTreeError::InvalidStructure(_))));
    }

    fn create_mixed_type_record_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float32, false), // float feature
            Field::new("f1", DataType::Int64, false),   // integer feature
            Field::new("f2", DataType::Boolean, false), // boolean feature
        ]);

        let float_array = Float32Array::from(vec![0.5, 0.3, 0.7, 0.4]);
        let int_array = Int64Array::from(vec![100, 50, 75, 25]);
        let bool_array = BooleanArray::from(vec![true, false, true, false]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(float_array),
                Arc::new(int_array),
                Arc::new(bool_array),
            ],
        )
        .unwrap()
    }

    fn create_mixed_type_tree() -> FeatureTree {
        // Create a tree that uses all feature types:
        //                [f0 < 0.5]
        //               /          \
        //        [f1 < 60]        [f2 == true]
        //        /       \        /           \
        //    [-1.0]    [0.0]  [1.0]        [2.0]

        FeatureTreeBuilder::new()
            .feature_names(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()])
            .feature_types(vec![
                FeatureType::Float,
                FeatureType::Int,
                FeatureType::Indicator,
            ])
            .split_indices(vec![0, 1, -1, -1, 2, -1, -1])
            .split_conditions(vec![0.5, 60.0, 0.0, 0.0, 0.5, 0.0, 0.0])
            .children(
                vec![1, 2, u32::MAX, u32::MAX, 5, u32::MAX, u32::MAX],
                vec![4, 3, u32::MAX, u32::MAX, 6, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 2.0])
            .default_left(vec![false, false, false, false, false, false, false, false])
            .build()
            .unwrap()
    }

    #[test]
    fn test_predict_arrays_mixed_types() {
        let tree = create_mixed_type_tree();
        let batch = create_mixed_type_record_batch();

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![tree],
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string(), "f2".to_string()]),
            feature_types: Arc::new(vec![
                FeatureType::Float,
                FeatureType::Int,
                FeatureType::Indicator,
            ]),
            base_score: 0.0,
            objective: Objective::SquaredError,
        };

        let predictions = gbdt.predict_arrays(batch.columns()).unwrap();

        // Row 0: f0=0.5, f1=100, f2=true(1.0)
        //   f0=0.5 >= 0.5 -> right path -> f2=1.0 >= 0.5 -> 2.0
        // Row 1: f0=0.3, f1=50, f2=false(0.0)
        //   f0=0.3 < 0.5 -> left path -> f1=50 < 60 -> -1.0
        // Row 2: f0=0.7, f1=75, f2=true(1.0)
        //   f0=0.7 >= 0.5 -> right path -> f2=1.0 >= 0.5 -> 2.0
        // Row 3: f0=0.4, f1=25, f2=false(0.0)
        //   f0=0.4 < 0.5 -> left path -> f1=25 < 60 -> -1.0

        let expected = [2.0, -1.0, 2.0, -1.0];
        for (i, &expected_value) in expected.iter().enumerate() {
            assert!(
                (predictions.value(i) - expected_value).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                predictions.value(i)
            );
        }
    }

    #[test]
    fn test_predict_arrays_batch_processing() {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float32, false),
            Field::new("f1", DataType::Float32, false),
        ]);

        let n_rows = 1000;
        let f0_data: Vec<f32> = (0..n_rows).map(|i| (i as f32) / n_rows as f32).collect();
        let f1_data: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 2.0).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float32Array::from(f0_data)),
                Arc::new(Float32Array::from(f1_data)),
            ],
        )
        .unwrap();

        let trees: Vec<FeatureTree> = (0..100) // goes into batch processing if trees>=100
            .map(|_| create_sample_tree())
            .collect();

        let gbdt = GradientBoostedDecisionTrees {
            trees,
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string()]),
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.0,
            objective: Objective::SquaredError,
        };

        let predictions = gbdt.predict_arrays(batch.columns()).unwrap();
        assert_eq!(predictions.len(), n_rows);
    }

    #[test]
    fn test_predict_arrays_error_handling() {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Utf8, false), // Unsupported type
            Field::new("f1", DataType::Float32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["invalid"])),
                Arc::new(Float32Array::from(vec![1.0])),
            ],
        )
        .unwrap();

        let gbdt = GradientBoostedDecisionTrees {
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["f0".to_string(), "f1".to_string()]),
            feature_types: Arc::new(vec![FeatureType::Float, FeatureType::Float]),
            base_score: 0.0,
            objective: Objective::SquaredError,
        };

        let result = gbdt.predict_arrays(batch.columns());
        assert!(matches!(result, Err(ArrowError::InvalidArgumentError(_))));
    }

    #[test]
    fn test_prune_with_default_direction_and_nulls() {
        // Create a deeper tree:
        //                    [feature0 < 0.5]
        //                   /               \
        //      [feature1 < 0.3]            [feature1 < 0.6]
        //     /               \            /               \
        // [feature2 < 0.7]    [-1.0]    [1.0]       [feature2 < 0.8]
        //   /        \                                /            \
        // [-2.0]    [2.0]                          [2.0]         [3.0]

        let feature_tree = FeatureTreeBuilder::new()
            .feature_names(vec![
                "feature0".to_string(),
                "feature1".to_string(),
                "feature2".to_string(),
            ])
            .feature_types(vec![
                FeatureType::Float,
                FeatureType::Float,
                FeatureType::Float,
            ])
            .split_indices(vec![0, 1, 2, -1, -1, 1, 2, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0])
            .children(
                vec![1, 3, 4, u32::MAX, u32::MAX, 7, 8, u32::MAX, u32::MAX],
                vec![5, 2, 6, u32::MAX, u32::MAX, 6, 8, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 2.0, 2.0, 3.0])
            .default_left(vec![
                true, false, false, false, false, false, false, false, false,
            ])
            .build()
            .unwrap();

        let predictions_right = feature_tree.predict(&[0.4, 0.3, 0.8]);

        let mut predicate1 = Predicate::new();
        predicate1.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.5));
        let pruned1 = feature_tree
            .prune(&predicate1, &["f0".to_string(), "f1".to_string()])
            .unwrap();
        let predicitons_after_pruning = pruned1.predict(&[f32::NAN, 0.3, 0.8]);
        assert_eq!(predictions_right, predicitons_after_pruning);
    }
}
