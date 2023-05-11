import os
from datetime import datetime
import numpy as np
from typing import Any, Callable
import joblib  # type: ignore
from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
    statistical_model,
)
from pipeline.models.utils.model_enum import Model
import pipeline.models.utils.scoring as scoring
import pandas as pd  # type: ignore

from pipeline.preprocessing.compute_features.feature import FeatureList, Feature
from pipeline.preprocessing.sk_formatter import SKFormatter

Params = dict[str, Any]
Models = dict[Model, Params]
Job = tuple[Model, Callable[[pd.DataFrame, pd.Series], tuple[Any, dict]]]


def runner(model_jobs: list[Job], formatter: SKFormatter) -> dict[str, pd.Series]:
    """
    The runner, at a high-level, is responsible for:
      1. Training the individual models of the model_jobs
      2. Save the SKFormatter params along side the models themselves, their params and metrics

    use joblib for saving models to file:
    # https://scikit-learn.org/stable/model_persistence.html

    Args:
        model_jobs (list[Job]):
        formatter (SKFormatter):

    returns:
        dict[str, pd.Series]: dict mapping model name to its predictions.
        The predictions can be indexed by osm_id.
    """
    date = datetime.today().strftime("%m_%d_%H_%M")
    path = f"/share-files/runs/{date}/{date}_"

    # Obtain train and test data
    x_train, _, y_train, _ = formatter.generate_train_test_split()
    _, x_test, _, y_test = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2013.pkl",
        test_size=1.0,
        discard_features=formatter.discard_features,
    ).generate_train_test_split()

    # Generate folders and save header for metrics
    metrics_file = f"{path}metrics"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "a+") as f:
        f.write("model,mae,mape,mse,rmse,r2,ev\n")  # header for metrics

    # Save SKFormatter params
    save_skformatter_params(formatter.params, path)

    # Train each model using gridsearch func defined in model_jobs list
    predictions: dict[str, pd.Series] = {}
    for model_name, model_func in model_jobs:
        # Train model, obtaining the best model and the corresponding hyper-parameters
        best_model, best_params = model_func(x_train, y_train)  # type: ignore

        # Get prediction and score model
        # predictions can be indexed with osm_id
        y = get_prediction(model_name.value, best_model, x_test)
        predictions[str(model_name.value)] = y
        metrics = scoring.score_model(y_test, y)

        # Save the model, hyper-parameters and metrics
        save_model_hyperparams_metrics(
            model_name.value, best_model, best_params, metrics, path
        )

    return predictions


def get_prediction(model_name: str, model: Model, x_test: pd.DataFrame) -> pd.Series:
    """
    Get a prediction based on the test-set provided.
    Args:
        model_name (str): The name of the model retrieving predictions for
        model (Model): The actual model, i.e. MLP, LogReg, XGB or RF
        x_test (pd.DataFrame): The test data to get predictions from

    Returns:
        pd.Series: A numpy array of predictions
    """
    if model_name in Model.regression_models_names():
        return scoring.classify_with_regressor(model, x_test)  # type: ignore
    else:
        return model.predict(x_test)  # type: ignore


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


def save_model_hyperparams_metrics(
    model_name: str, model: Model, params: dict, metrics: dict[str, float], prefix: str
):
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
    save_model(model_name, model, prefix)
    save_params(model_name, params, prefix)
    save_metrics(model_name, metrics, prefix)


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
    with open(file, "a+") as f:
        f.write(f"\nmodel: {model_name}, params: {params}")


def save_metrics(model_name: str, metrics: dict[str, float], filepath: str) -> None:
    """
    Saves the metrics dict to disk in the filepath location.
    Args:
        model_name (str): Name of the model currently being saved
        metrics (dict[str, float]): The dictionary of metrics to save
        filepath (str): The filepath location for saving the file
    """
    file = filepath + "metrics"
    with open(file, "a+") as f:
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
        (
            Model.STATMODEL,
            statistical_model,
        ),  # should work now, since input is a dataframe
    ]

    formatter = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2012.pkl",
        test_size=0.0,
        # Patch with desired features for models
        discard_features=FeatureList(
            [
                Feature.OSM_ID,
                Feature.COORDINATES,
                Feature.CPR_VEJNAVN,
                Feature.HAST_SENEST_RETTET,
                Feature.DISTANCES,
            ]
        ),
    )
    runner(model_jobs, formatter)


if __name__ == "__main__":
    main()
