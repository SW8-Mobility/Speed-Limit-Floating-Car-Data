from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from bisect import bisect_left

SPEED_LIMITS = [15, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]


def find_closest_speed_limit(speed: float) -> float:
    """
    Returns closest value in speed_limits.
    If two numbers are equally close, return the smallest number - this could potentially be changed in the future.
    """
    index = bisect_left(SPEED_LIMITS, speed)
    if index == 0:
        return SPEED_LIMITS[0]
    if index == len(SPEED_LIMITS):
        return SPEED_LIMITS[-1]
    before = SPEED_LIMITS[index - 1]
    after = SPEED_LIMITS[index]
    if after - speed < speed - before:
        return after
    else:
        return before


def classify_with_regressor(model, x: pd.DataFrame) -> list[float]:
    """calls predict on the model and quantizes the result.

    Args:
        model (Model): Model with a predict method
        x (pd.Series): the features to predict with

    Returns:
        list[float]: list of predictions quantized
    """
    predictions = model.predict(x)
    return quantize_results(predictions)


def quantize_results(predictions: np.ndarray) -> list[float]:
    """Snaps each prediction to the closest speed limit.

    Args:
        predictions (np.ndarray): list of predictions made by regression model.

    Returns:
        list[float]: each prediction in the list quantized
    """
    return [find_closest_speed_limit(x) for x in predictions]


def mean_absolute_percentage_error(ground_truth, prediction) -> float:
    """Computes mean absolute percentage error

    Args:
        ground_truth (Array-like object): ground truth
        prediction (Array-like object): prediction made by regression / prediction model

    Returns:
        float: mean absolute percentage error
    """
    y_true, y_pred = np.array(ground_truth), np.array(prediction)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # type: ignore


def score_model(ground_truth, prediction) -> dict[str, float]:
    """Scoring function with metrics used for regression models. Will compute:
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_error,
    r2, and explained variance.

    Args:
        ground_truth (Array-like object): ground truth
        prediction (Array-like object): prediction made by a model

    Returns:
        dict[str, float]: dict of metrics to their result
    """
    return {
        "mae": mean_absolute_error(ground_truth, prediction),
        "mape": mean_absolute_percentage_error(ground_truth, prediction),
        "mse": mean_squared_error(ground_truth, prediction),
        "rmse": mean_squared_error(ground_truth, prediction, squared=False),
        "r2": r2_score(ground_truth, prediction),
        "ev": explained_variance_score(ground_truth, prediction),
    }  # type: ignore
