use pyo3::prelude::*;

pub mod arch;
pub mod loader;
pub mod objective;
pub mod predicates;
mod python;
pub mod tree;

pub use loader::ModelLoader;
pub use objective::Objective;
pub use predicates::{Condition, Predicate};
pub use tree::{FeatureTree, FeatureTreeBuilder, GradientBoostedDecisionTrees};

#[pymodule]
fn trusty(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let internal = py.import_bound("trusty._internal")?;
    m.add("_internal", internal)?;
    Ok(())
}

#[pymodule]
fn _internal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(python::load_model))?;
    m.add_class::<python::PyGradientBoostedDecisionTrees>()?;
    m.add_class::<python::Feature>()?;
    Ok(())
}
