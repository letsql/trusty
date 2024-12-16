import json
import pandas as pd
import numpy as np
import pyarrow as pa
import trusty

from trusty import Feature
from pathlib import Path


TEST_DIR = Path(__file__).parent.parent.parent


def test_predict():
    df = pd.read_csv(
        TEST_DIR/"tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    )
    with open(
        TEST_DIR/"tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json", "r"
    ) as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)
    model = trusty.load_model(model_json_str)
    actual_preds = df['prediction'].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    batch = pa.RecordBatch.from_pandas(df)
    predictions = model.predict_batches([batch])
    assert len(predictions) == len(df) 
    np.testing.assert_array_almost_equal(np.array(predictions), np.array(actual_preds), decimal=3)


def test_pruning():
    df = pd.read_csv(
        TEST_DIR/"tests/data/reg:squarederror/diamonds_data_filtered_trees_100_mixed.csv"
    ).query("carat <0.2")
    with open(
        TEST_DIR/"tests/models/reg:squarederror/diamonds_model_trees_100_mixed.json", "r"
    ) as f:
        model_json = json.load(f)
        model_json_str = json.dumps(model_json)
    model = trusty.load_model(model_json_str)
    batch = pa.RecordBatch.from_pandas(df)
    predicates = [Feature("carat") < 0.2]
    actual_preds = df['prediction'].copy().to_list()
    df = df.drop(["target", "prediction"], axis=1)
    pruned_model = model.prune(predicates)
    predictions = pruned_model.predict_batches([batch])
    assert len(predictions) == len(df)
    assert all([isinstance(p, float) for p in predictions])
    assert all([p >= 0 for p in predictions])
    np.testing.assert_array_almost_equal(np.array(predictions), np.array(actual_preds), decimal=3) # 10^(-12)
