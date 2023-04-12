from typing import Any
import pandas as pd
from bisect import bisect_left
from sklearn.base import BaseEstimator
from pipeline.preprocessing.compute_features.feature import Feature
SPEED_LIMITS = [15,30,40,50,60,70,80,90,100,110,120,130]

def find_closest_speed_limit(speed: float) -> float:
    """
    Returns closest value in speed_limits.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(SPEED_LIMITS, speed)
    if pos == 0:
        return SPEED_LIMITS[0]
    if pos == len(SPEED_LIMITS):
        return SPEED_LIMITS[-1]
    before = SPEED_LIMITS[pos - 1]
    after = SPEED_LIMITS[pos]
    if after - speed < speed - before:
        return after
    else:
        return before

def qunatize_results(predictions):
    return [find_closest_speed_limit(pred) for pred in predictions]