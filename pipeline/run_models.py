import numpy as np
from typing import Any, Callable
import joblib
from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
    statistical_model,
)
from pipeline.models.utils.model_enum import Model
import pipeline.models.utils.scoring as scoring
from pipeline.models.models import (
    create_mlp_grid_search,
    random_forest_regressor_gridsearch,
    xgboost_classifier_gridsearch,
    logistic_regression_gridsearch,
)
from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences

pd.options.display.width = 0

# Model = dict[str, Model]
Params = dict[str, Any]
Models = dict[Model, Params]

def generate_x(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.drop([Feature.OSM_ID.value], axis=1)
    for feature in Feature.array_2d_features():
        df[feature] = df[feature].apply(lambda row: sum(row, []))

    for feature in Feature.array_1d_features() + Feature.array_2d_features():
        df[feature] = df[feature].apply(lambda arr: [elem for elem in arr if arr is not None])
        df[feature] = pad_sequences(df[feature], padding='post').tolist()
        df[feature] = df[feature].apply(lambda arr: np.array(arr))

    xs = [df[f].values.tolist() for f in df.columns]
    x = np.concatenate(xs, axis=1)

    return x

def prepare_df_for_training(
    df_feature_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df: pd.DataFrame = pd.read_pickle(df_feature_path).head(150)

    df = df.drop( # some of these should be one hot encoded instead of dropped
        columns=[
            Feature.COORDINATES.value,
            Feature.CPR_VEJNAVN.value,
            Feature.HAST_SENEST_RETTET.value,
            Feature.VEJSTIKLASSE.value,
            Feature.VEJTYPESKILTET.value,
        ]
    )

    df = df.rename(columns={"hast_gaeldende_hast": "target"})

    encode_single_value_features(df)

    y = df["target"].values
    df = df.drop(["target"], axis=1)

    x = generate_x(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return df, x_train, x_test, y_train, y_test


def encode_single_value_features(df):
    for f in Feature.numeric_features():
        df[f] = df[f].apply(lambda val: np.array([val]))


def train_models_save_results(
    x_train, y_train
) -> dict[Model, Any]:  # TODO: update docstring
    """
    Creates every model from models.py, fits them, saves
    them to pickle, saves best params and returns dict
    mapping model names to the fitted model.

    use joblib for saving models to file:
    # https://scikit-learn.org/stable/model_persistence.html

    Args:
        x_train: Training dataset
        y_train: Target for training

    Returns: dictionary of model name to trained model model
    """

    # define a list of models and their corresponding grid search functions (from models.py)
    model_jobs = [
        # (Model.MLP, create_mlp_grid_search),
        # (Model.RF, random_forest_regressor_gridsearch),
        # (Model.XGB, xgboost_classifier_gridsearch),
        (Model.LOGREG, logistic_regression_gridsearch),
        # (Model.STATMODEL, statistical_model),
    ]

    models: dict[str, Any] = {}  # model name to the trained model

    with open("models/training_results.txt", "a") as best_model_params_f:
        # loop through each model and perform grid search
        for model_name, model_func in model_jobs:
            best_model, best_params = model_func(x_train, y_train)
            best_model_params_f.write(  # save the best params to file
                f"\nmodel: {model_name.value}, params: {best_params}"
            )
            joblib.dump(  # save the model as joblib file
                best_model, f"{model_name.value}_best_model.joblib"
            )
            models[model_name] = best_model

    return models


def test_models(
    models: dict[Model, Any], x_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Tests all the models. Will return scoring metrics for each models predictions.

    Args:
        models (dict[Model, Any]): The dictionary of the best models after fitting on the train data.
        x_test (pd.DataFrame): The input test data from the train-test split
        y_test (pd.Series): The target test data from the train-test split
    """
    scored_predictions = pd.DataFrame({"y_true": y_test})
    # initialize scored_predictions with y_test
    for model_name, model in models.items():
        if model_name.value in Model.regression_models_names():
            y_pred = scoring.classify_with_regressor(model, x_test)
        else:
            y_pred = model.predict(x_test)
        scores = scoring.score_model(y_test, y_pred)
        scored_predictions[
            f"{model_name}_y_pred"
        ] = y_pred  # add a new column for each model's predictions

        for score_name, score_value in scores.items():
            scored_predictions[
                f"{model_name}_{score_name}"
            ] = score_value  # add a new column for each score

    return scored_predictions


def main():
    df, x_train, x_test, y_train, y_test = prepare_df_for_training(
        "/share-files/pickle_files_features_and_ground_truth/2012.pkl"
    )
    models = train_models_save_results(x_train, y_train)
    scores = test_models(models, x_test, y_test)
    scores.to_csv("/share-files/model_scores/scores.csv", index=False)

if __name__ == "__main__":
    df, x_train, x_test, y_train, y_test = prepare_df_for_training(
        "/share-files/pickle_files_features_and_ground_truth/2012.pkl"
    )
    train_models_save_results(x_train, y_train)