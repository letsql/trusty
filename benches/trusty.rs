#![allow(unused_must_use)]
pub mod common;
use arrow::array::{Array, Float32Array};
use arrow::compute::concat;
use arrow::record_batch::RecordBatch;
use common::data_loader;
use criterion::{criterion_group, criterion_main, Criterion};
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;
use rayon::prelude::*;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use tokio::runtime::Runtime;
use trusty::loader::ModelLoader;
use trusty::predicates::{Condition, Predicate};
use trusty::tree::GradientBoostedDecisionTrees;

const BATCHSIZE: usize = 8192;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn predict_batch(
    trees: &GradientBoostedDecisionTrees,
    batches: &[RecordBatch],
) -> Result<Float32Array> {
    trees
        .predict_batches(batches)
        .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)
}

fn predict_batch_with_gbdt(model: &GBDT, batches: &[RecordBatch]) -> Result<Float32Array> {
    let predictions: Vec<Float32Array> = batches
        .par_iter()
        .map(|batch| -> Result<Float32Array> {
            let mut result = Vec::new();
            for row in 0..batch.num_rows() {
                let mut row_data = Vec::new();
                for col in batch.columns() {
                    if let Some(array) = col.as_any().downcast_ref::<Float32Array>() {
                        row_data.push(array.value(row).into());
                    }
                }
                result.push(Data::new_test_data(row_data, None));
            }
            let predictions = model.predict(&result);
            Ok(Float32Array::from(Vec::from_iter(
                predictions.into_iter().map(|x| x as f32),
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    let arrays_ref: Vec<&dyn Array> = predictions.iter().map(|a| a as &dyn Array).collect();
    let concatenated =
        concat(&arrays_ref).map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    Ok(concatenated
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            Box::<dyn Error + Send + Sync>::from("Failed to downcast concatenated array")
        })?
        .clone())
}

fn benchmark_diamonds_prediction(c: &mut Criterion) -> Result<()> {
    let rt = Runtime::new()?;
    let trees =
        load_model("data/benches/reg:squarederror/models/diamonds_model_trees_100_mixed.json")?;
    let (data_batches, _) = data_loader::load_diamonds_dataset(
        "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_mixed.csv",
        BATCHSIZE,
        false,
    )?;
    let baseline_predictions = predict_batch(&trees, &data_batches)?;
    let total_rows: usize = data_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        baseline_predictions.len(),
        total_rows,
        "Predictions length {} doesn't match total rows {}",
        baseline_predictions.len(),
        total_rows
    );

    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition("carat".to_string(), Condition::LessThan(0.3));
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    let pruned_predictions = predict_batch(&pruned_trees, &data_batches)?;
    assert_eq!(
        pruned_predictions.len(),
        total_rows,
        "Pruned predictions length {} doesn't match total rows {}",
        pruned_predictions.len(),
        total_rows
    );

    c.bench_function("trusty/diamonds/baseline", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    c.bench_function("trusty/diamonds/manual_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    Ok(())
}

fn benchmark_airline_prediction(c: &mut Criterion) -> Result<()> {
    let rt = Runtime::new()?;
    let trees = load_model(
        "data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_mixed.json",
    )?;
    let (data_batches, _) = data_loader::load_airline_dataset(
        "data/benches/reg:squarederror/data/airline_satisfaction_data_filtered_trees_1000_mixed.csv",
        BATCHSIZE,
        false,
    )?;
    let predicate = {
        let mut pred = Predicate::new();
        pred.add_condition(
            "online_boarding".to_string(),
            Condition::GreaterThanOrEqual(4.0),
        );
        pred
    };
    let pruned_trees = trees.prune(&predicate);

    let mut group = c.benchmark_group("trusty/airline");

    group.bench_function("baseline", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&trees, &data_batches).unwrap() })
    });

    group.bench_function("manual_pruning", |b| {
        b.to_async(&rt)
            .iter(|| async { predict_batch(&pruned_trees, &data_batches).unwrap() })
    });

    group.finish();
    Ok(())
}

fn benchmark_implementations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    {
        let trees = load_model(
            "data/benches/reg:squarederror/models/diamonds_model_trees_100_float64.json",
        )
        .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_diamonds_dataset(
            "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let trusty_predictions = predict_batch(&trees, &batches)
            .expect("Failed to generate trusty predictions for diamonds");
        assert_eq!(
            trusty_predictions.len(),
            total_rows,
            "Trusty diamonds predictions length {} doesn't match total rows {}",
            trusty_predictions.len(),
            total_rows
        );

        c.bench_function("trusty/diamonds/float64", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch(&trees, &batches).unwrap() })
        });
    }
    {
        let trees = load_model("data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_float64.json")
            .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_airline_dataset(
            "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let trusty_predictions = predict_batch(&trees, &batches)
            .expect("Failed to generate trusty predictions for diamonds");
        assert_eq!(
            trusty_predictions.len(),
            total_rows,
            "Trusty diamonds predictions length {} doesn't match total rows {}",
            trusty_predictions.len(),
            total_rows
        );

        c.bench_function("trusty/airline/float64", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch(&trees, &batches).unwrap() })
        });
    }

    {
        let model = GBDT::from_xgboost_json_used_feature("data/benches/reg:squarederror/models/airline_satisfaction_model_trees_1000_float64.json")
            .expect("Failed to load airline model");
        let (batches, _) = data_loader::load_airline_dataset(
            "data/benches/reg:squarederror/data/airline_satisfaction_data_full_trees_1000_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load airline data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let gbdt_predictions = predict_batch_with_gbdt(&model, &batches)
            .expect("Failed to generate GBDT predictions for airline");
        assert_eq!(
            gbdt_predictions.len(),
            total_rows,
            "GBDT airline predictions length {} doesn't match total rows {}",
            gbdt_predictions.len(),
            total_rows
        );

        c.bench_function("gbdt/airline", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch_with_gbdt(&model, &batches).unwrap() })
        });
    }

    {
        let model = GBDT::from_xgboost_json_used_feature(
            "data/benches/reg:squarederror/models/diamonds_model_trees_100_float64.json",
        )
        .expect("Failed to load diamonds model");
        let (batches, _) = data_loader::load_diamonds_dataset(
            "data/benches/reg:squarederror/data/diamonds_data_full_trees_100_float64.csv",
            BATCHSIZE,
            true,
        )
        .expect("Failed to load diamonds data");

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        let gbdt_predictions = predict_batch_with_gbdt(&model, &batches)
            .expect("Failed to generate GBDT predictions for diamonds");
        assert_eq!(
            gbdt_predictions.len(),
            total_rows,
            "GBDT diamonds predictions length {} doesn't match total rows {}",
            gbdt_predictions.len(),
            total_rows
        );

        c.bench_function("gbdt/diamonds", |b| {
            b.to_async(&rt)
                .iter(|| async { predict_batch_with_gbdt(&model, &batches).unwrap() })
        });
    }
}

fn load_model(path: &str) -> Result<GradientBoostedDecisionTrees> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model_data: Value = serde_json::from_reader(reader)?;
    Ok(GradientBoostedDecisionTrees::load_from_json(&model_data)?)
}

criterion_group! {
    name = trusty;
    config = Criterion::default();
    targets =
        benchmark_diamonds_prediction,
        benchmark_airline_prediction,
        benchmark_implementations
}

criterion_main!(trusty);
