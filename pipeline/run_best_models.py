import os
from datetime import datetime
from typing import Callable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.pipeline import Pipeline

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

Model = tuple[Model, Callable[[pd.DataFrame, pd.Series], Pipeline]]


def runner(model_jobs: list[Model], formatter: SKFormatter) -> None:
    """
    The runner, at a high-level, is responsible for:
      1. Fitting the individual models of the model_jobs
      2. Save the SKFormatter params along side the metrics of each model

    Args:
        model_jobs (list[Model]): Models for which to fit using existing best params
        formatter (SKFormatter): SKFormatter formatting the training sets.
    """

    # Obtain train and test data
    x_train, _, y_train, _ = formatter.generate_train_test_split()
    _, x_test, _, y_test = SKFormatter(
        "/share-files/pickle_files_features_and_ground_truth/2013.pkl",
        test_size=1.0,
        discard_features=formatter.discard_features,
    ).generate_train_test_split()

    date = datetime.today().strftime("%m_%d_%H_%M")
    path = f"/share-files/runs/{date}/{date}_"

    # Generate folders and save header for metrics
    metrics_file = f"{path}metrics"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "a+") as f:
        f.write("model,mae,mape,mse,rmse,r2,ev\n")  # header for metrics

    # Save SKFormatter params
    save_skformatter_params(formatter.params, f"{path}_skf_params")

    # Test each of the best_models using params from earlier run
    for model_name, pipeline in model_jobs:
        # Fit model
        model = pipeline(x_train, y_train)

        # Test model
        y_pred = get_prediction(model_name.value, model, x_test)  # type: ignore

        # Save metrics
        metrics = scoring.score_model(y_test, y_pred)
        save_metrics(model_name.value, metrics, metrics_file)


def main():
    # define a list of models and their corresponding model using best params
    model_jobs: list[Model] = [
        (Model.MLP, create_mlp_best_params),
        (Model.RF, create_rf_best_params),
        (Model.XGB, create_xgboost_best_params),
        (Model.LOGREG, create_logistic_regression_best_params),
        # (Model.STATMODEL, statistical_model), # TODO: We do not have a StatModel yet
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

    # Run new GridSearch
    run_models.main()


if __name__ == "__main__":
    main()
