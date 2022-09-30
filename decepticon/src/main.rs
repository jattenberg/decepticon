use ferris_says::say;
use std::io::{stdout, BufWriter};

use smartcore::dataset::*;
// DenseMatrix wrapper around Vec
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
// Imports for KNN classifier
use smartcore::neighbors::knn_classifier::KNNClassifier;
// Model performance
use smartcore::metrics::roc_auc_score;
use smartcore::model_selection::train_test_split;

fn main() {
    let stdout = stdout();

    let mut writer = BufWriter::new(stdout.lock());

    let cancer_data = breast_cancer::load_dataset();
    // Transform dataset into a NxM matrix
    let x = DenseMatrix::from_array(
        cancer_data.num_samples,
        cancer_data.num_features,
        &cancer_data.data,
    );
    // These are our target class labels
    let y = cancer_data.target;
    // Split dataset into training/test (80%/20%)
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);
    // KNN classifier
    let y_hat_knn = KNNClassifier::fit(
        &x_train,
        &y_train,        
        Default::default(),
    ).and_then(|knn| knn.predict(&x_test)).unwrap();    

    // Calculate test error

    let score = roc_auc_score(&y_test, &y_hat_knn);
    let message = format!("AUC: {score}", );
    let width = message.chars().count();
    say(message.as_bytes(), width, &mut writer).unwrap();
}
