from typing import Any, Callable

from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
    statistical_model,
)
from pipeline.models.utils.model_enum import Model
from pipeline.models.utils.scoring import score_model
from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
)
from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd

pd.options.display.width = 0

Model = dict[str, Model]
Params = dict[str, Any]
Models = dict[Model, Params]

import joblib

def get_fake_input():
    """
    Simple placeholder function for returning dummy input.
    Returns:
        pd.DataFrame: A dataframe containing fake/arbitrary input
    """
    columns = ["target"]
    for feature in Feature:
        if (
            feature != Feature.ROAD_TYPE
            and feature != Feature.ID
            and feature != Feature.COORDINATES
            and feature != Feature.LEVEL
            and feature != Feature.DAY_OF_WEEK
            and feature != Feature.TIME_GROUP
            and feature != Feature.VCR
        ):
            columns.append(feature.value)

    input_df = pd.DataFrame(
        data=[
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ],
        ],
        columns=columns,
    )

    return input_df

def prepare_df_for_training(df_feature_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    input_df: pd.DataFrame = pd.read_pickle(df_feature_path)

    x = input_df.drop(columns=["hast_gaeldende_hast"])
    y = input_df["hast_gaeldende_hast"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    df = df.rename(columns={"hast_gaeldende_hast": "target"})

    return df, x_train, x_test, y_train, y_test

def train_models_save_results(x_train, y_train) -> dict[str, Any]: #TODO: update docstring
    """

    Args:
        x_train:
        y_train:

    Returns: dictionary with key being the model name and value being a model (object)

    """

    # define a list of models and their corresponding grid search functions (from models.py)
    model_jobs = [
        (Model.MLP, create_mlp_grid_search),
        (Model.RF, random_forest_regressor_gridsearch),
        (Model.XGB, xgboost_classifier_gridsearch),
        (Model.LOGREG, logistic_regression_gridsearch),
        (Model.STATMODEL, statistical_model),
    ]

    models = {}

    with open("training_results.txt", "a") as best_model_params_f:
        # loop through each model and perform grid search
        for model_name, model_func in model_jobs:
            best_model, best_params = model_func(x_train, y_train)
            best_model_params_f.write(f"model: {model_name.value}, params: {best_params}")
            joblib.dump(best_model, f'{model_name.value}_best_model.joblib') # https://scikit-learn.org/stable/model_persistence.html
            models[model_name.value] = best_model

    return models

def test_models(models: Models, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Tests all the models based on obtained best models.
    Args:
        models (Models): The dictionary of the best models after fitting on the train data.
        x_test (pd.DataFrame): The input test data from the train-test split
        y_test (pd.Series): The target test data from the train-test split
    """
    scored_predictions = pd.DataFrame(
        {"y_true": y_test}
    )
    # initialize scored_predictions with y_test
    for model_name, model_info in models.items():
        model = model_info["model"]
        y_pred = model.predict(x_test)
        scores = score_model(y_test, y_pred)
        scored_predictions[
            f"{model_name}_y_pred"
        ] = y_pred  # add a new column for each model's predictions
        for score_name, score_value in scores.items():
            scored_predictions[
                f"{model_name}_{score_name}"
            ] = score_value  # add a new column for each score

    return scored_predictions


def main():
    df, x_train, x_test, y_train, y_test = prepare_df_for_training("/share-files/2012_with_ground.pkl")
    train_models_save_results(x_train, y_train)


if __name__ == "__main__":
    main()
