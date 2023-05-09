from datetime import datetime
import numpy as np
from typing import Any, Callable
import joblib  # type: ignore
from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
)
from pipeline.models.utils.model_enum import Model
import pipeline.models.utils.scoring as scoring
import pandas as pd  # type: ignore
from pipeline.preprocessing.sk_formatter import SKFormatter

Params = dict[str, Any]
Models = dict[Model, Params]
Job = tuple[Model, Callable[[pd.DataFrame, pd.DataFrame], tuple[Any, dict]]]


def runner(model_jobs: list[Job], formatter: SKFormatter) -> None:
    """
    The runner, at a high-level, is responsible for:
      1. Training the individual models of the model_jobs
      2. Save the SKFormatter params along side the models themselves, their params and metrics

    use joblib for saving models to file:
    # https://scikit-learn.org/stable/model_persistence.html

    Args:
        model_jobs (list[Job]):
        formatter (SKFormatter):
    """
    prefix = datetime.today().strftime("%m%d_%H%M_")

    x_train, x_test, y_train, y_test = formatter.generate_train_test_split()
    
    save_skformatter_params(formatter.params, prefix)

    file = prefix + "metrics"
    with open(file, "a") as f:
        f.write("model,mae,mape,mse,rmse,r2,ev\n")  # header for metrics

    # Train each model using gridsearch func defined in model_jobs list
    for model_name, model_func in model_jobs:
        # Train model, obtaining the best model and the corresponding hyper-parameters
        best_model, best_params = model_func(x_train, y_train)  # type: ignore

        # Get prediction and score model
        y_pred = get_prediction(model_name.value, best_model, x_test)
        append_predictions_to_df(df, y_pred, model_name)  # type: ignore
        metrics = scoring.score_model(y_test, y_pred)

        # Save the model, hyper-parameters and metrics
        save(model_name, best_model, best_params, metrics, prefix)


def get_prediction(model_name: str, model: Model, x_test: np.ndarray) -> np.ndarray:
    """
    Get a prediction based on the test-set provided
    Args:
        model_name (str): The name of the model retrieving predictions for
        model (Model): The actual model, i.e. MLP, LogReg, XGB or RF
        x_test (np.ndarray): The test data to get predictions from

    Returns:
        np.ndarray: A numpy array of predictions
    """
    if model_name in Model.regression_models_names():
        return scoring.classify_with_regressor(model, x_test)  # type: ignore
    else:
        return model.predict(x_test)


def append_predictions_to_df(
    df: pd.DataFrame, predictions: np.ndarray, model: Model
) -> pd.DataFrame:
    """Save predictions made to dataframe.

    Args:
        df (pd.DataFrame): Dataframe to append predictions to.
        predictions (np.ndarray): Predictions made by a model
        model (Model): Which model made the predictions

    Returns:
        pd.DataFrame: Annoted dataframe
    """

    # The first test_size number of rows are used for testing
    # ex. 1000 row dataframe might have a test_size of 200 rows
    # to append the 200 rows of predictions, padding is needed
    # so that it is the same length.
    # so, pad to same length and append.
    predictions_padded = np.pad(
        predictions,
        (0, len(df) - len(predictions)),
        mode="constant",  # type: ignore
        constant_values=-1,  # pad with -1
    )
    col_name = f"{model.value}_preds"
    df[col_name] = pd.Series(predictions_padded)

    # do not keep rows, where no predictions were made
    df = df[df[col_name] != -1]

    return df


def save(model_name: str, model: Model, params: dict, metrics: dict[str, float], prefix: str):
    """
    Saves 3 files:
      1. The model,
      2. Hyper-parameters, and
      3. Metrics
    Args:
        model_name (str): Name of the model currently being saved
        model (Model): The model currently being saved
        params (dict): The dict of params to save
        metrics (dict[str, float]): The dictionary of metrics to save
        prefix (str): The folder location for saving the files

    Returns:

    """
    filepath = f"{prefix}/{prefix}_"
    save_model(model_name, model, filepath)
    save_params(model_name, params, filepath)
    save_metrics(model_name, metrics, filepath)


def save_model(model_name: str, model: Model, filepath: str) -> None:
    """
    Saves the model to disk in the filepath location as a joblib file
    Args:
        model_name (str): Name of the model currently being saved
        model (Model): The model currently being saved
        filepath (str): The filepath location for saving the file
    """
    joblib.dump(  # save the model as joblib file
        model, f"{filepath}{model_name}.joblib"
    )


def save_params(model_name: str, params: dict, filepath: str) -> None:
    """
    Saves the params to disk in the filepath location
    Args:
        model_name (str): Name of the model currently being saved
        params (dict): The dict of params to save
        filepath (str): The filepath location for saving the file
    """
    file = filepath + "params"
    with open(file, "a") as f:
        f.write(
            f"\nmodel: {model_name}, params: {params}"
            )


def save_metrics(model_name: str, metrics: dict[str, float], filepath: str) -> None:
    """
    Saves the metrics dict to disk in the filepath location.
    Args:
        model_name (str): Name of the model currently being saved
        metrics (dict[str, float]): The dictionary of metrics to save
        filepath (str): The filepath location for saving the file
    """
    file = filepath + "metrics"
    with open(file, "a") as f:
        f.write(f"{model_name}")
        for val in metrics.values():
            f.write(f", {val}")
        f.write("\n")

        
def save_skformatter_params(params: dict, filepath: str) -> None:
    """Save the skf parameters from model training to a file.
    Will name file based on time created.

    Args:
        params (dict): parameters from SKFormatter
        filepath (str): Filepath for file
    """
    filename = f"{filepath}skf_parameters.txt"

    with open(filename, "w+") as f:
        f.write(f'{", ".join(params.keys())}\n')  # header
        for param, value in params.items():
            f.write(f"{param}")
            f.write(f", {value}")
            f.write("\n")


def main():
    # define a list of models and their corresponding grid search functions (from models.py)
    model_jobs: list[Job] = [
        (Model.MLP, create_mlp_grid_search),
        (Model.RF, random_forest_regressor_gridsearch),
        (Model.XGB, xgboost_classifier_gridsearch),
        (Model.LOGREG, logistic_regression_gridsearch),
        # (Model.STATMODEL, statistical_model), # TODO: Does not work currently...
    ]

    formatter = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2012.pkl"
    )
    runner(model_jobs, formatter)


if __name__ == "__main__":
    main()
