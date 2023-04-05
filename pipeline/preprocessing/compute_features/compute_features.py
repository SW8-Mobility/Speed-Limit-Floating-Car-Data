import sys
from os.path import dirname, realpath

current_dir = dirname(realpath(__file__))
parentdir = dirname(dirname(dirname(current_dir)))
sys.path.append(parentdir)

from statistics import mean, median
from typing import Any, Callable, Union
from pipeline.preprocessing.compute_features.feature import Feature
import pandas as pd #type: ignore
from functools import partial
from calculate_speeds_distances import calculate_speeds, calculate_distances


def none_if_empty(func: Callable, input: list[float]) -> Union[float, None]:
    """Will return None if input for function is empty, otherwise result of function.

    Args:
        func (Callable): function to call
        input (list[float]): input for function, always list

    Returns:
        Union[float, None]: None or the result of the function call
    """
    if len(input) == 0:
        return None
    else:
        return func(input)


def per_trip_computation(func: Callable, row: pd.DataFrame) -> list[float]:
    """
    Calls the given function on the speed column in the row.

    Args:
        func: a function to call on speed column
        row: a row from formatted dataframe

    Returns: results from input function when applied on the speed coloumn

    """

    return [
        func(speed_list)
        for speed_list in row[Feature.SPEEDS.value]
        if len(speed_list) > 0
    ]

def aggregate_results(
    aggregate_func: Callable, feature: Feature, row: pd.DataFrame
) -> Union[None, float]:
    """
    Aggregate results in a given feature coloumn with input aggregate function (e.g. mean).

    Args:
        aggregate_func: the function to a
        feature: the row name to be aggregated
        row: a row from formatted dataframe

    Union[float, None]: None or the result of the function call

    """
    results = row[feature.value]
    return none_if_empty(aggregate_func, results)  # type: ignore

def compute_distances(row: pd.DataFrame) -> list[list[float]]:
    """
    Compute the distance between each point for all trips.

    Args:
        row: a row from a formatted dataframe

    Returns: list of (list of distances), e.g. for a given osm id, for each trip through it, there will be a list of
             distances.

    """
    return [calculate_distances(trip) for trip in row[Feature.COORDINATES.value]]


def compute_speeds(row: pd.DataFrame) -> list[list[float]]:
    """
    Computes the speeds for a given row. The "distances" column must be computed first!

    Args:
        row: a row from formatted dataframe

    Returns: list of [list of speeds], e.g. for a given osm id, for each trip through it, there will be a list of speeds

    """
    return [
        calculate_speeds(trip, dist)
        for dist, trip in zip(
            row[Feature.DISTANCES.value], row[Feature.COORDINATES.value]
        )
    ]


def add_features_to_df(df: pd.DataFrame) -> None:
    """
    Add features to formatted dataframe.

    Args:
        df: a formatted dataframe, where an osm id has a number of trips

    """
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
        partial(per_trip_computation, min),
        partial(per_trip_computation, max),
        partial(per_trip_computation, mean),
        partial(per_trip_computation, median),
        partial(aggregate_results, mean, Feature.MINS),
        partial(aggregate_results, mean, Feature.MAXS),
        partial(aggregate_results, mean, Feature.MEANS),
        partial(aggregate_results, mean, Feature.MEDIANS),
    ]

    for feature_name, func in zip(features, feature_calculators):
        df[feature_name.value] = df.apply(func, axis=1)


def main():
    path = "C:/Users/ax111/Documents/Personal documents/Coding/SW8/speed_limit_floating_car_data/pipeline/preprocessing/compute_features/test.pkl"
    df: pd.DataFrame = pd.read_pickle(path).head(1000)
    add_features_to_df(df)
    df.drop([Feature.COORDINATES.value], inplace=True, axis=1)
    df.to_csv("test_features.csv")


if __name__ == "__main__":
    main()
