from typing import Any
import pandas as pd
from bisect import bisect_left
from sklearn.base import BaseEstimator

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from a CSV file and split it into features and target.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of two dataframes: the features and target.
    """
    df = pd.read_csv(path)
    y = df['speed_limit']
    X = df.drop(columns=['speed_limit'])
    return X, y

def f1_score(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate a MLPClassifier classifier on the testing data using F1-score.

    Args:
        model (sklearn classifier): The model to evaluate.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        float: The F1-score of the model on the testing data.
    """
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='micro') # type: ignore

def find_closest_speed_limit(speed: float) -> float:
    """
    Returns closest value in speed_limits.
    If two numbers are equally close, return the smallest number.
    """
    speed_limits = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    pos = bisect_left(speed_limits, speed)
    if pos == 0:
        return speed_limits[0]
    if pos == len(speed_limits):
        return speed_limits[-1]
    before = speed_limits[pos - 1]
    after = speed_limits[pos]
    if after - speed < speed - before:
        return after
    else:
        return before