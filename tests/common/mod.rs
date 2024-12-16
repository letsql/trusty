use arrow::array::{ArrayRef, BooleanArray, Float32Array, Int64Array};
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use prettytable::{format, Cell, Row, Table};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use trusty::loader::ModelLoader;
use trusty::GradientBoostedDecisionTrees;

pub fn read_diamonds_csv_to_split_batches(
    path: &str,
    batch_size: usize,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
    let file = File::open(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("carat", DataType::Float32, false),
        Field::new("depth", DataType::Float32, true),
        Field::new("table", DataType::Float32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("cut_good", DataType::Boolean, false),
        Field::new("cut_ideal", DataType::Boolean, false),
        Field::new("cut_premium", DataType::Boolean, false),
        Field::new("cut_very_good", DataType::Boolean, false),
        Field::new("color_e", DataType::Boolean, false),
        Field::new("color_f", DataType::Boolean, false),
        Field::new("color_g", DataType::Boolean, false),
        Field::new("color_h", DataType::Boolean, false),
        Field::new("color_i", DataType::Boolean, false),
        Field::new("color_j", DataType::Boolean, false),
        Field::new("clarity_if", DataType::Boolean, false),
        Field::new("clarity_si1", DataType::Boolean, false),
        Field::new("clarity_si2", DataType::Boolean, false),
        Field::new("clarity_vs1", DataType::Boolean, false),
        Field::new("clarity_vs2", DataType::Boolean, false),
        Field::new("clarity_vvs1", DataType::Boolean, false),
        Field::new("clarity_vvs2", DataType::Boolean, false),
        Field::new("target", DataType::Int64, false),
        Field::new("prediction", DataType::Float32, false),
    ]));

    let csv = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(batch_size)
        .build(file)?;

    let batches: Vec<_> = csv.collect::<Result<_, _>>()?;

    let feature_schema = Arc::new(Schema::new(schema.fields()[0..23].to_vec()));
    let target_prediction_schema = Arc::new(Schema::new(schema.fields()[23..].to_vec()));

    let mut feature_batches = Vec::new();
    let mut target_prediction_batches = Vec::new();

    for batch in batches {
        let feature_columns: Vec<ArrayRef> = batch.columns()[0..23].to_vec();
        let target_prediction_columns: Vec<ArrayRef> = batch.columns()[23..].to_vec();

        let feature_batch = RecordBatch::try_new(feature_schema.clone(), feature_columns)?;
        let target_prediction_batch =
            RecordBatch::try_new(target_prediction_schema.clone(), target_prediction_columns)?;

        feature_batches.push(feature_batch);
        target_prediction_batches.push(target_prediction_batch);
    }

    Ok((feature_batches, target_prediction_batches))
}

pub fn read_airline_csv_to_split_batches(
    path: &str,
    batch_size: usize,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
    let file = File::open(path)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("gender", DataType::Int64, false),
        Field::new("customer_type", DataType::Int64, false),
        Field::new("age", DataType::Int64, false),
        Field::new("type_of_travel", DataType::Int64, false),
        Field::new("class", DataType::Int64, false),
        Field::new("flight_distance", DataType::Int64, false),
        Field::new("inflight_wifi_service", DataType::Int64, false),
        Field::new("departure/arrival_time_convenient", DataType::Int64, false),
        Field::new("ease_of_online_booking", DataType::Int64, false),
        Field::new("gate_location", DataType::Int64, false),
        Field::new("food_and_drink", DataType::Int64, false),
        Field::new("online_boarding", DataType::Int64, false),
        Field::new("seat_comfort", DataType::Int64, false),
        Field::new("inflight_entertainment", DataType::Int64, false),
        Field::new("on_board_service", DataType::Int64, false),
        Field::new("leg_room_service", DataType::Int64, false),
        Field::new("baggage_handling", DataType::Int64, false),
        Field::new("checkin_service", DataType::Int64, false),
        Field::new("inflight_service", DataType::Int64, false),
        Field::new("cleanliness", DataType::Int64, false),
        Field::new("departure_delay_in_minutes", DataType::Int64, false),
        Field::new("arrival_delay_in_minutes", DataType::Float32, false),
        Field::new("target", DataType::Int64, false),
        Field::new("prediction", DataType::Float32, false),
    ]));

    let csv = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(batch_size)
        .build(file)?;

    let batches: Vec<_> = csv.collect::<Result<_, _>>()?;

    let feature_schema = Arc::new(Schema::new(schema.fields()[0..23].to_vec()));
    let target_prediction_schema = Arc::new(Schema::new(schema.fields()[23..].to_vec()));

    let mut feature_batches = Vec::new();
    let mut target_prediction_batches = Vec::new();

    for batch in batches {
        let feature_columns: Vec<ArrayRef> = batch.columns()[0..23].to_vec();
        let target_prediction_columns: Vec<ArrayRef> = batch.columns()[23..].to_vec();

        let feature_batch = RecordBatch::try_new(feature_schema.clone(), feature_columns)?;
        let target_prediction_batch =
            RecordBatch::try_new(target_prediction_schema.clone(), target_prediction_columns)?;

        feature_batches.push(feature_batch);
        target_prediction_batches.push(target_prediction_batch);
    }

    Ok((feature_batches, target_prediction_batches))
}

pub struct PredictionComparator {
    epsilon: f32,
}

impl PredictionComparator {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
    pub fn compare_predictions(
        &self,
        trusty_predictions: &[Float32Array],
        expected_predictions: &[&Float32Array],
        preprocessed_batches: &[RecordBatch],
        expected_results: &[RecordBatch],
    ) -> Result<(), Box<dyn Error>> {
        for (batch_idx, ((trusty, expected), (preprocessed_batch, expected_batch))) in
            trusty_predictions
                .iter()
                .zip(expected_predictions)
                .zip(preprocessed_batches.iter().zip(expected_results))
                .enumerate()
        {
            self.validate_batch_predictions(
                batch_idx,
                trusty,
                expected,
                preprocessed_batch,
                expected_batch,
            )?;
        }
        Ok(())
    }

    fn validate_batch_predictions(
        &self,
        batch_idx: usize,
        trusty: &Float32Array,
        expected: &Float32Array,
        preprocessed_batch: &RecordBatch,
        expected_batch: &RecordBatch,
    ) -> Result<(), Box<dyn Error>> {
        if trusty.len() != expected.len() {
            return Err(format!(
                "Batch {}: Prediction arrays have different lengths - trusty: {}, expected: {}",
                batch_idx,
                trusty.len(),
                expected.len()
            )
            .into());
        }

        let differences = self.collect_differences(
            batch_idx,
            trusty,
            expected,
            preprocessed_batch,
            expected_batch,
        );

        if !differences.is_empty() {
            self.print_differences(batch_idx, &differences);
            return Err(format!(
                "Batch {}: Found {} predictions that differ by more than epsilon ({})",
                batch_idx,
                differences.len(),
                self.epsilon
            )
            .into());
        }

        Ok(())
    }

    fn collect_differences(
        &self,
        _batch_idx: usize,
        trusty: &Float32Array,
        expected: &Float32Array,
        preprocessed_batch: &RecordBatch,
        expected_batch: &RecordBatch,
    ) -> Vec<PredictionDifference> {
        let mut differences = Vec::new();

        for (idx, (t, e)) in trusty.iter().zip(expected.iter()).enumerate() {
            if let (Some(t_val), Some(e_val)) = (t, e) {
                if (t_val - e_val).abs() > self.epsilon {
                    let feature_values =
                        self.collect_feature_values(idx, preprocessed_batch, expected_batch);
                    differences.push(PredictionDifference {
                        index: idx,
                        trusty_value: t_val,
                        expected_value: e_val,
                        features: feature_values,
                    });
                }
            }
        }

        differences
    }

    fn collect_feature_values(
        &self,
        idx: usize,
        preprocessed_batch: &RecordBatch,
        expected_batch: &RecordBatch,
    ) -> Vec<FeatureComparison> {
        let mut all_columns = std::collections::HashSet::new();
        for col_idx in 0..preprocessed_batch.num_columns() {
            all_columns.insert(
                preprocessed_batch
                    .schema()
                    .field(col_idx)
                    .name()
                    .to_string(),
            );
        }
        for col_idx in 0..expected_batch.num_columns() {
            all_columns.insert(expected_batch.schema().field(col_idx).name().to_string());
        }

        all_columns
            .into_iter()
            .map(|col_name| {
                let preprocessed_value = preprocessed_batch
                    .column_by_name(&col_name)
                    .map(|col| get_value_at_index(col, idx))
                    .unwrap_or_else(|| "N/A".to_string());

                let expected_value = expected_batch
                    .column_by_name(&col_name)
                    .map(|col| get_value_at_index(col, idx))
                    .unwrap_or_else(|| "N/A".to_string());

                let col_type = preprocessed_batch
                    .column_by_name(&col_name)
                    .map(|col| col.data_type().to_string())
                    .or_else(|| {
                        expected_batch
                            .column_by_name(&col_name)
                            .map(|col| col.data_type().to_string())
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                FeatureComparison {
                    name: col_name,
                    preprocessed_value,
                    expected_value,
                    feature_type: col_type,
                }
            })
            .collect()
    }

    fn print_differences(&self, batch_idx: usize, differences: &[PredictionDifference]) {
        println!("\nBatch {} - Failed Predictions:", batch_idx);

        for difference in differences {
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_BOX_CHARS);

            difference.print_to_table(&mut table);
            table.printstd();
            println!("\n");
        }
    }
}

struct PredictionDifference {
    index: usize,
    trusty_value: f32,
    expected_value: f32,
    features: Vec<FeatureComparison>,
}

impl PredictionDifference {
    fn print_to_table(&self, table: &mut Table) {
        table.add_row(Row::new(vec![
            Cell::new("Index"),
            Cell::new(&self.index.to_string()),
            Cell::new(""),
            Cell::new("Type"),
        ]));
        table.add_row(Row::new(vec![
            Cell::new("Trusty Prediction"),
            Cell::new(&format!("{:.6}", self.trusty_value)),
            Cell::new(""),
            Cell::new("Float32"),
        ]));
        table.add_row(Row::new(vec![
            Cell::new("Expected Prediction"),
            Cell::new(&format!("{:.6}", self.expected_value)),
            Cell::new(""),
            Cell::new("Float32"),
        ]));
        table.add_row(Row::new(vec![
            Cell::new("Difference"),
            Cell::new(&format!(
                "{:.6}",
                (self.trusty_value - self.expected_value).abs()
            )),
            Cell::new(""),
            Cell::new(""),
        ]));

        table.add_row(Row::new(vec![
            Cell::new("Feature"),
            Cell::new("Original Value"),
            Cell::new("Test Value"),
            Cell::new("Type"),
        ]));

        for feature in &self.features {
            feature.print_to_table(table);
        }
    }
}

struct FeatureComparison {
    name: String,
    preprocessed_value: String,
    expected_value: String,
    feature_type: String,
}

impl FeatureComparison {
    fn print_to_table(&self, table: &mut Table) {
        table.add_row(Row::new(vec![
            Cell::new(&self.name),
            Cell::new(&self.preprocessed_value),
            Cell::new(&self.expected_value),
            Cell::new(&self.feature_type),
        ]));
    }
}

fn get_value_at_index(array: &ArrayRef, idx: usize) -> String {
    if array.is_null(idx) {
        return "null".to_string();
    }

    if let Some(float_array) = array.as_any().downcast_ref::<Float32Array>() {
        return format!("{:.6}", float_array.value(idx));
    }

    if let Some(bool_array) = array.as_any().downcast_ref::<BooleanArray>() {
        return bool_array.value(idx).to_string();
    }

    if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
        return int_array.value(idx).to_string();
    }

    format!("unsupported type: {}", array.data_type())
}
pub struct ModelTester {
    epsilon: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatasetType {
    Diamonds,
    Airline,
}

impl ModelTester {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    pub fn test_model(
        &self,
        model_path: &str,
        test_data_path: &str,
        batch_size: usize,
        dataset_type: DatasetType,
    ) -> Result<(), Box<dyn Error>> {
        let trees = self.load_model(model_path)?;
        let (preprocessed_batches, expected_results) =
            self.load_dataset(test_data_path, batch_size, dataset_type)?;

        let expected_predictions = self.extract_expected_predictions(&expected_results)?;
        let trusty_predictions = self.generate_predictions(&trees, &preprocessed_batches)?;

        let comparator = PredictionComparator::new(self.epsilon);
        comparator.compare_predictions(
            &trusty_predictions,
            &expected_predictions,
            &preprocessed_batches,
            &expected_results,
        )?;

        println!("All predictions match!");
        Ok(())
    }

    pub fn load_dataset(
        &self,
        data_path: &str,
        batch_size: usize,
        dataset_type: DatasetType,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>), Box<dyn Error>> {
        match dataset_type {
            DatasetType::Diamonds => read_diamonds_csv_to_split_batches(data_path, batch_size),
            DatasetType::Airline => read_airline_csv_to_split_batches(data_path, batch_size),
        }
    }

    pub fn load_model(
        &self,
        model_path: &str,
    ) -> Result<GradientBoostedDecisionTrees, Box<dyn Error>> {
        let model_data: Value = serde_json::from_reader(BufReader::new(File::open(model_path)?))?;

        Ok(GradientBoostedDecisionTrees::load_from_json(&model_data)?)
    }

    pub fn extract_expected_predictions<'a>(
        &self,
        expected_results: &'a [RecordBatch],
    ) -> Result<Vec<&'a Float32Array>, Box<dyn Error>> {
        expected_results
            .iter()
            .map(|batch| {
                batch
                    .column_by_name("prediction")
                    .ok_or_else(|| "Column 'prediction' not found".into())
                    .and_then(|col| {
                        col.as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| "Failed to downcast to Float32Array".into())
                    })
            })
            .collect()
    }

    fn generate_predictions(
        &self,
        trees: &GradientBoostedDecisionTrees,
        preprocessed_batches: &[RecordBatch],
    ) -> Result<Vec<Float32Array>, Box<dyn Error>> {
        preprocessed_batches
            .iter()
            .map(|batch| {
                trees
                    .predict_batches(&[batch.clone()])
                    .map_err(|e| Box::new(e) as Box<dyn Error>)
            })
            .collect()
    }
}
