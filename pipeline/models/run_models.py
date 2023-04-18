from typing import Any, Callable

from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch, statistical_model,
)
from pipeline.models.utils.model_enum import ModelEnum
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


Model = dict[str, ModelEnum]
Params = dict[str, Any]
Models = dict[tuple[Model, Params]]


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


def train_models():
    """
    Runs all models and records the results in the results dictionary.
    Also prints the best estimators and their parameters.
    """
    input_df = get_fake_input()

    x = input_df.drop(columns=["target"])
    y = input_df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # define a list of models and their corresponding grid search functions (from models.py)
    models = [
        (ModelEnum.MLP, create_mlp_grid_search),
        (ModelEnum.RF, random_forest_regressor_gridsearch),
        (ModelEnum.XGB, xgboost_classifier_gridsearch),
        (ModelEnum.LOGREG, logistic_regression_gridsearch),
        (ModelEnum.STATMODEL, statistical_model),
    ]

    results = {}

    # loop through each model and perform grid search
    for model_name, model_func in models:
        if model_name == ModelEnum.STATMODEL:  # Handle statistical model separately
            best_model = model_func()
            results[model_name] = {"model": best_model, "params": None}
        else:
            best_model, best_params = model_func(x_train, y_train)
            results[model_name] = {"model": best_model, "params": best_params}

    test_models(results, x_test, y_test)


def test_models(
    models: Models, x_test: pd.DataFrame, y_test: pd.Series
):
    """
    Tests all the models based on obtained best models.
    Args:
        models (Models): The dictionary of the best models after fitting on the train data.
        x_test (pd.DataFrame): The input test data from the train-test split
        y_test (pd.Series): The target test data from the train-test split
    """
    scored_predictions = pd.DataFrame({'y_true': y_test})  # initialize scored_predictions with y_test
    for model_name, model_info in models.items():
        model = model_info["model"]
        y_pred = model.predict(x_test)
        scores = score_model(y_test, y_pred)
        scored_predictions[f"{model_name}_y_pred"] = y_pred  # add a new column for each model's predictions
        for score_name, score_value in scores.items():
            scored_predictions[f"{model_name}_{score_name}"] = score_value  # add a new column for each score

    return scored_predictions




def main():
    train_models()


if __name__ == "__main__":
    main()
