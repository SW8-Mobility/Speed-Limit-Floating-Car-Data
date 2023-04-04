import sys
from os.path import dirname, realpath
current_dir = dirname(realpath(__file__))
parentdir = dirname(dirname(dirname(current_dir)))
sys.path.append(parentdir)

from statistics import mean, median
from typing import Any, Callable, Union
from pipeline.preprocessing.compute_features.feature import Feature
import pandas as pd
from functools import partial
from pipeline.preprocessing.compute_features.type_alias import ListOfSpeeds, Trip, FeatureCalculator
from calculate_speeds_distances import calculate_speeds, calculate_distances

def func_default_resturn(func: Callable, input: list[float]) -> Union[float, None]:
    if len(input) == 0:
        return None
    else:
        return func(input)

def compute_min_speeds(row: pd.DataFrame) -> list[float]:
    return [min(speed_list) for speed_list in row[Feature.SPEEDS.value] if len(speed_list) > 0]

def compute_max_speeds(row: pd.DataFrame) -> list[float]:
    return [max(speed_list) for speed_list in row[Feature.SPEEDS.value] if len(speed_list) > 0]

def compute_median_speeds(row: pd.DataFrame) -> list[float]:
    return [median(speed_list) for speed_list in row[Feature.SPEEDS.value] if len(speed_list) > 0]

def compute_mean_speeds(row: pd.DataFrame) -> list[float]:
    return [mean(speed_list) for speed_list in row[Feature.SPEEDS.value] if len(speed_list) > 0]

def compute_aggregate_max(row: pd.DataFrame) -> Union[None, float]:
    maxs = row[Feature.MAXS.value]
    return func_default_resturn(max, maxs) # type: ignore

def compute_aggregate_mean(row: pd.DataFrame) -> Union[None, float]:
    means = row[Feature.MEANS.value]
    return func_default_resturn(mean, means) # type: ignore

def compute_aggregate_min(row: pd.DataFrame) -> Union[None, float]:
    mins = row[Feature.MINS.value]
    return func_default_resturn(min, mins) # type: ignore

def compute_aggregate_median(row: pd.DataFrame) -> Union[None, float]:
    medians = row[Feature.MEDIANS.value]
    return func_default_resturn(median, medians) # type: ignore

def compute_distances(row: pd.DataFrame) -> list[list[float]]:
    return [calculate_distances(trip) for trip in row[Feature.COORDINATES.value]]

def compute_speeds(row: pd.DataFrame) -> list[list[float]]:
    return [calculate_speeds(trip, dist) for dist, trip in zip(row[Feature.DISTANCES.value], row[Feature.COORDINATES.value])]

def add_features_to_df(df: pd.DataFrame) -> None:
    features = [
        Feature.DISTANCES,
        Feature.SPEEDS,
        Feature.MINS,
        Feature.MAXS,
        Feature.MEANS,
        Feature.MEDIANS,
        Feature.AGGREGATE_MIN,
        Feature.AGGREGATE_MAX,
        Feature.AGGREGATE_MEAN,
        Feature.AGGREGATE_MEDIAN,
    ]
    feature_calculators = [
        compute_distances,
        compute_speeds,
        compute_min_speeds,
        compute_max_speeds,
        compute_mean_speeds,
        compute_median_speeds,
        compute_aggregate_min,
        compute_aggregate_max,
        compute_aggregate_mean,
        compute_aggregate_median
    ]

    for feature_name, func in zip(features, feature_calculators):
        df[feature_name.value] = df.apply(func, axis=1)


def main():
    path = 'C:/Users/ax111/Documents/Personal documents/Coding/SW8/speed_limit_floating_car_data/pipeline/preprocessing/compute_features/test.pkl'
    df: pd.DataFrame = pd.read_pickle(path).head(1000)
    add_features_to_df(df)
    df.drop([Feature.COORDINATES.value], inplace=True, axis=1)
    df.to_csv("test_features.csv")

if __name__ == '__main__':
    main()