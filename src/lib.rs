pub mod loader;
pub mod objective;
pub mod predicates;
pub mod tree;
pub use objective::Objective;
pub use predicates::{AutoPredicate, Condition, Predicate};
pub use tree::{FeatureTree, FeatureTreeBuilder, GradientBoostedDecisionTrees};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::array::Float64Array;
    use arrow::array::Float64Builder;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn create_sample_tree() -> FeatureTree {
        let feature_names = vec!["feature0".to_string()];
        let feature_types = vec!["float".to_string()];

        FeatureTreeBuilder::new()
            .feature_names(feature_names)
            .feature_types(feature_types)
            .split_indices(vec![
                0,  // root: split on feature0
                -1, // left leaf
                -1, // right leaf
            ])
            .split_conditions(vec![
                0.5, // root: split at 0.5
                0.0, // left leaf (unused)
                0.0, // right leaf (unused)
            ])
            // children: u32::MAX indicates no children (leaf)
            .children(
                vec![1, u32::MAX, u32::MAX], // left children
                vec![2, u32::MAX, u32::MAX], // right children
            )
            .base_weights(vec![
                0.0,  // root (internal nodes have weight 0)
                -1.0, // left leaf prediction
                1.0,  // right leaf prediction
            ])
            .build()
            .expect("Valid tree definition")
    }

    fn create_tree_nested_features() -> FeatureTree {
        //              feature0 < 1.0
        //             /             \
        //    feature0 < 0.5         Leaf (2.0)
        //   /           \
        // Leaf (-1.0)  Leaf (1.0)
        FeatureTreeBuilder::new()
            .feature_names(vec!["feature0".to_string(), "feature1".to_string()])
            .feature_types(vec!["float".to_string(), "float".to_string()])
            .split_indices(vec![0, 0, -1, -1, -1])
            .split_conditions(vec![1.0, 0.5, 0.0, 0.0, 0.0])
            .children(
                vec![1, 3, u32::MAX, u32::MAX, u32::MAX],
                vec![2, 4, u32::MAX, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, 2.0, -1.0, 1.0])
            .build()
            .unwrap()
    }

    fn create_sample_tree_deep() -> FeatureTree {
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
            .split_indices(vec![0, 1, 2, -1, -1, -1, -1])
            .split_conditions(vec![0.5, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0])
            .children(
                vec![1, 3, 5, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
                vec![2, 4, 6, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            )
            .base_weights(vec![0.0, 0.0, 0.0, -2.0, -1.0, 1.0, 2.0])
            .build()
            .unwrap()
    }

    #[test]
    fn test_tree_predict() {
        let tree = create_sample_tree();
        assert_eq!(tree.predict(&[0.4]), -1.0);
        assert_eq!(tree.predict(&[0.6]), 1.0);
    }

    #[test]
    fn test_tree_depth() {
        let tree = create_sample_tree();
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_tree_prune() {
        let tree = create_sample_tree();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.49));
        let pruned_tree = tree.prune(&predicate, &["feature0".to_string()]).unwrap();
        assert_eq!(pruned_tree.tree.len(), 1);
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
        predicate2.add_condition("feature2".to_string(), Condition::LessThan(0.69));
        let pruned_tree2 = tree.prune(&predicate2, &feature_names).unwrap();
        assert_eq!(pruned_tree2.predict(&[0.4, 0.6, 0.8]), -1.0);

        // Test case 3: Prune left root tree
        let mut predicate3 = Predicate::new();
        predicate3.add_condition("feature0".to_string(), Condition::GreaterThanOrEqual(0.50));
        let pruned_tree3 = tree.prune(&predicate3, &feature_names).unwrap();
        assert_eq!(pruned_tree3.predict(&[0.4, 0.6, 0.8]), 2.0);
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
        assert_eq!(pruned_tree.predict(&[0.4, 0.0, 1.0]), 2.0);

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        predicate.add_condition("feature2".to_string(), Condition::GreaterThanOrEqual(0.7));

        let pruned_tree = tree.prune(&predicate, &feature_names).unwrap();
        assert_eq!(pruned_tree.predict(&[0.6, 0.3, 0.5]), -1.0);
        assert_eq!(pruned_tree.predict(&[0.8, 0.29, 1.0]), -2.0);
    }

    #[test]
    fn test_trees_predict_batch() {
        let trees = GradientBoostedDecisionTrees {
            base_score: 0.5,
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let schema = Schema::new(vec![Field::new("feature0", DataType::Float64, false)]);
        let feature_data = Float64Array::from(vec![0.4, 0.6]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature_data)]).unwrap();

        let predictions = trees.predict_batch(&batch).unwrap();
        assert_eq!(predictions.value(0), -0.5); // 0.5 (base_score) + -1.0
        assert_eq!(predictions.value(1), 1.5); // 0.5 (base_score) + 1.0
    }

    #[test]
    fn test_trees_predict_batch_with_missing_values() {
        let trees = GradientBoostedDecisionTrees {
            base_score: 0.5,
            trees: vec![create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let schema = Schema::new(vec![Field::new("feature0", DataType::Float64, true)]);
        let feature_data = Float64Array::from(vec![Some(0.4), Some(0.6), None, Some(0.5)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(feature_data)]).unwrap();

        let predictions = trees.predict_batch(&batch).unwrap();
        assert_eq!(predictions.value(2), -0.5);
    }

    #[test]
    fn test_trees_num_trees() {
        let trees = GradientBoostedDecisionTrees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };
        assert_eq!(trees.num_trees(), 2);
    }

    #[test]
    fn test_trees_tree_depths() {
        let trees = GradientBoostedDecisionTrees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };
        assert_eq!(trees.tree_depths(), vec![2, 2]);
    }

    #[test]
    fn test_trees_prune() {
        let trees = GradientBoostedDecisionTrees {
            base_score: 0.5,
            trees: vec![create_sample_tree(), create_sample_tree()],
            feature_names: Arc::new(vec!["feature0".to_string()]),
            feature_types: Arc::new(vec!["float".to_string()]),
            objective: Objective::SquaredError,
        };

        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.50));

        let pruned_trees = trees.prune(&predicate);
        assert_eq!(pruned_trees.trees.len(), 2);
        assert_eq!(pruned_trees.trees[0].tree.len(), 1);
        assert_eq!(pruned_trees.trees[1].tree.len(), 1);
    }

    #[test]
    fn test_trees_nested_features() {
        let tree = create_tree_nested_features();
        let mut predicate = Predicate::new();
        predicate.add_condition("feature0".to_string(), Condition::LessThan(0.4));
        let pruned_tree = tree
            .prune(
                &predicate,
                &["feature0".to_string(), "feature1".to_string()],
            )
            .unwrap();

        assert_eq!(tree.predict(&[0.3, 0.0]), -1.0);
        assert_eq!(tree.predict(&[0.7, 0.0]), 1.0);
        assert_eq!(tree.predict(&[1.5, 0.0]), 2.0);

        assert_eq!(pruned_tree.predict(&[0.3, 0.0]), -1.0);
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
}
