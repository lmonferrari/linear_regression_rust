use linfa::dataset::DatasetBase;
use linfa::traits::{Fit, Predict};
use linfa_linear::LinearRegression;
use ndarray::array;

fn mean_squared_error(y_true: &ndarray::Array1<f64>, y_pred: &ndarray::Array1<f64>) -> f64 {
    let diff = y_true - y_pred;

    match diff.mapv(|x| x.powi(2)).mean() {
        Some(x) => x,
        None => panic!("mean squared error could not be computed"),
    }
}

fn main() {
    // data to datset
    let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0];
    let dataset = DatasetBase::new(x.clone(), y.clone());

    // creting model
    let model = LinearRegression::default();

    //model fit
    let model = model.fit(&dataset).expect("Error traning model");

    // test x and y
    let x_test = array![[5.2, 6.1], [6.3, 7.4]];
    let y_test = array![11.0, 13.0];

    // prediction
    let y_pred = model.predict(&x_test);
    let y_pred_rounded = y_pred.mapv(|x: f64| x.round());
    println!("Predictions : {}", y_pred_rounded);

    // calculate the error
    let mse = mean_squared_error(&y_test, &y_pred_rounded);
    println!("Calculed mse: {}", mse);
}
