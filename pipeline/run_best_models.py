import os
from datetime import datetime
from typing import Callable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from pipeline import run_models
from pipeline.models.best_params_models import (
    create_mlp_best_params,
    create_rf_best_params,
    create_xgboost_best_params,
    create_logistic_regression_best_params,
)
from pipeline.models.utils import scoring
from pipeline.models.utils.model_enum import Model
from pipeline.preprocessing.compute_features.feature import FeatureList, Feature

from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.run_models import get_prediction, save_metrics, save_skformatter_params

Model_best = tuple[Model, Callable[[pd.DataFrame, pd.Series], Pipeline]]


def runner(
    model_jobs: list[Model_best],
    train_formatter: SKFormatter,
    test_formatter: SKFormatter,
) -> dict[str, pd.Series]:
    """
    The runner, at a high-level, is responsible for:
      1. Fitting the individual models of the model_jobs
      2. Save the SKFormatter params along side the metrics of each model

    Args:
        model_jobs (list[Model]): Models for which to fit using existing best params
        train_formatter (SKFormatter): SKFormatter instance for formatting the training set
        test_formatter (SKFormatter): SKFormatter instance for formatting the test set
    Returns:
        dict[str, pd.Series]: dict mapping model name to its predictions.
        The predictions can be indexed by osm_id.
    """

    # Obtain train and test data
    x_train, _, y_train, _ = train_formatter.generate_train_test_split()
    _, x_test, _, y_test = test_formatter.generate_train_test_split()

    date = datetime.today().strftime("%m_%d_%H_%M")
    path = f"/share-files/runs/{date}/{date}_"

    # Generate folders and save header for metrics
    metrics_file = f"{path}metrics"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "a+") as f:
        f.write("model,mae,mape,mse,rmse,r2,ev\n")  # header for metrics

    # Save SKFormatter params
    save_skformatter_params(train_formatter.params, f"{path}_skf_params")

    # Test each of the best_models using params from earlier run
    predictions: dict[str, pd.Series] = {}
    for model_name, pipeline in model_jobs:
        # Fit model
        model = pipeline(x_train, y_train)

        # Test model
        y = get_prediction(model_name.value, model, x_test)  # type: ignore
        predictions[str(model_name.value)] = y

        # Save metrics
        metrics = scoring.score_model(y_test, y)
        save_metrics(model_name.value, metrics, metrics_file)


def main():
    # define a list of models and their corresponding model using best params
    model_jobs: list[Model_best] = [
        (Model.MLP, create_mlp_best_params),
        (Model.RF, create_rf_best_params),
        (Model.XGB, create_xgboost_best_params),
        (Model.LOGREG, create_logistic_regression_best_params),
        # (Model.STATMODEL, statistical_model), # TODO: We do not have a StatModel yet
    ]

    train_format = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2012.pkl",
        test_size=0.0,
        discard_features=FeatureList(
            [
                Feature.OSM_ID,
                Feature.COORDINATES,
                Feature.DISTANCES,
            ]
        ),
    )

    test_format = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2013.pkl",
        test_size=1.0,
        discard_features=train_format.discard_features,
    )

    runner(model_jobs, train_format, test_format)

    # Run new GridSearch
    run_models.main()


if __name__ == "__main__":
    main()
