use crate::loader::ModelLoader;
use crate::tree::GradientBoostedDecisionTrees;
use crate::Condition;
use crate::Predicate;
use arrow::array::ArrayRef;
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Feature {
    name: String,
}

#[pymethods]
impl Feature {
    #[new]
    fn new(name: &str) -> Self {
        Feature {
            name: name.to_string(),
        }
    }

    fn __lt__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), false, other) // false means LessThan
    }

    fn __ge__(&self, other: f64) -> (String, bool, f64) {
        (self.name.clone(), true, other) // true means GreaterThanOrEqual
    }
}

#[pyclass]
pub struct PyGradientBoostedDecisionTrees {
    model: GradientBoostedDecisionTrees,
}

#[pymethods]
impl PyGradientBoostedDecisionTrees {
    #[new]
    fn new(model_json: &str) -> PyResult<Self> {
        let model_data: serde_json::Value = serde_json::from_str(model_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let model = GradientBoostedDecisionTrees::load_from_json(&model_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyGradientBoostedDecisionTrees { model })
    }

    fn predict_batches(&self, py_record_batches: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let mut batches = Vec::with_capacity(py_record_batches.len());

        for py_batch in py_record_batches.iter() {
            let py_arrow_type = py_batch.extract::<PyArrowType<RecordBatch>>()?;
            let record_batch = py_arrow_type.0;

            // convert Float64 arrays to Float32
            let arrays: Vec<ArrayRef> = record_batch
                .columns()
                .iter()
                .map(|col| {
                    if col.data_type() == &DataType::Float64 {
                        cast(col, &DataType::Float32).unwrap()
                    } else {
                        Arc::clone(col)
                    }
                })
                .collect();

            let new_schema = Schema::new(
                record_batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| {
                        if field.data_type() == &DataType::Float64 {
                            Arc::new(Field::new(
                                field.name(),
                                DataType::Float32,
                                field.is_nullable(),
                            ))
                        } else {
                            field.clone()
                        }
                    })
                    .collect::<Vec<Arc<Field>>>(),
            );

            let float32_batch = RecordBatch::try_new(Arc::new(new_schema), arrays).unwrap();
            batches.push(float32_batch);
        }

        // Process all batches at once
        let result = self
            .model
            .predict_batches(&batches)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(result.values().to_vec())
    }

    fn prune(&self, predicates: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut predicate = Predicate::new();
        for pred in predicates.iter() {
            let (feature_name, is_gte, threshold): (String, bool, f64) = pred.extract()?;
            let condition = if is_gte {
                Condition::GreaterThanOrEqual(threshold)
            } else {
                Condition::LessThan(threshold)
            };
            predicate.add_condition(feature_name, condition);
        }
        Ok(Self {
            model: self.model.prune(&predicate),
        })
    }

    fn print_tree_info(&self) {
        self.model.print_tree_info();
    }
}

#[pyfunction]
pub fn load_model(model_json: &str) -> PyResult<PyGradientBoostedDecisionTrees> {
    PyGradientBoostedDecisionTrees::new(model_json)
}
