from pipeline.models.models import create_mlp_grid_search, random_forest_regressor_gridsearch, \
    xgboost_classifier_gridsearch, logistic_regression_gridsearch
from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd

pd.options.display.width = 0


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
            ], [
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
            ], [
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
            ], [
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
            ], [
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
            ], [
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
            ], [
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
            ], [
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
            ], [
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
            ]
        ],
        columns=columns,
    )

    return input_df


def run_models():
    """
    Runs all models and records the results in the results dictionary.
    Also prints the best estimators and their parameters.
    """
    input_df = get_fake_input()

    X = input_df.drop(columns=["target"])
    y = input_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define a list of models and their corresponding grid search functions (from models.py)
    models = [("mlp", create_mlp_grid_search),
              ("random forest", random_forest_regressor_gridsearch),
              ("xgboost", xgboost_classifier_gridsearch),
              ("logistic regression", logistic_regression_gridsearch)]

    results = {}

    # loop through each model and perform grid search
    for model_name, model_func in models:
        best_model, best_params = model_func(X_train, y_train)
        results[model_name] = {"model": best_model, "params": best_params}

        # print the best parameters for each model
        print(f"Best parameters for {model_name}: {best_params}")


def main():
    run_models()


if __name__ == "__main__":
    main()
