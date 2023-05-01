""" Some functions used to inspect and analysize our data.
"""

from functools import reduce
from math import sqrt
from itertools import tee
from statistics import median, mean
import sys
from typing import Iterator
import pandas as pd  # type: ignore
from pipeline.preprocessing.df_processing import (
    create_df_from_json,
)


def copy_iterable(iter: Iterator) -> Iterator:
    """Returns a copy of an iterator. Does not modify the original

    Args:
        iter (Iterator): the iterator to copy

    Returns:
        Iterator: a copy of the Iterator.
    """
    return tee(iter, 1)[0]  # tee returns a tuple of 1 elemt


def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate aggregate min, max, and avg for dataframe

    Args:
        df (pd.DataFrame): Dataframe to calculate on.
                           Dataframe must have 'speeds' column.

    Returns:
        tuple[float, float, float]: tuple containing the avg, min and max speed
    """
    # per entry: max, min, avg speed across entire segment, median
    df["avg_speed"] = df["speeds"].apply(mean)  # type: ignore
    df["max_speed"] = df["speeds"].apply(max)
    df["min_speed"] = df["speeds"].apply(min)
    df["median"] = df["speeds"].apply(median)  # type: ignore

    # for all entries: median_avg, median_min, median_max
    median_avg = df["avg_speed"].median()
    median_min = df["min_speed"].median()
    median_max = df["max_speed"].median()

    return (median_avg, median_min, median_max)


def verify_solution(l: Iterator[bool]):
    """Auxiliary fuction used in filter_segments, to verify the solution works.
    I assume that trips do not loop back and go through the same
    segment more than once. This function is a sanitity check for this.

    Args:
        l (Iterator[bool]): list with boolean mask of all coordinates that are in
        the segment of interest.
    """
    cmp_if_different = lambda b, acc: True if len(acc) == 0 else acc[-1] != b
    reduced: Iterator[
        bool
    ] = reduce(  # convert [False, False, True, True, True, False] -> [False, True, False]
        lambda acc, bool: acc + [bool] if cmp_if_different(bool, acc) else acc, list(l), []  # type: ignore
    )

    # if there are more than one true in list
    # then the car has looped and break assumption
    if len(list(filter(lambda e: e is True, reduced))) > 1:
        print("Bad assumption...")
        sys.exit()


def filter_segments(df: pd.DataFrame, osm_id: int) -> pd.DataFrame:
    """Discard all rows from coordinates that do not correspond to the given osm_id.

    Args:
        df (pd.DataFrame): geodata dataframe, must have coordinates and osm_id cols
        osm_id (int): segment id

    Returns:
        pd.DataFrame: dataframe only with coordinates corresponding to given osm_id
    """

    def filter_func(row):
        """Used for filtering in df.apply and is called upon every row in the dataframe"""
        valid_osm_ids: Iterator[
            bool
        ] = map(  # id mask corresponding to which coordinates to keep
            lambda elem: elem == osm_id, row["osm_id"]
        )
        verify_solution(copy_iterable(valid_osm_ids))

        valid_coordinates = list(
            map(
                lambda tup: tup[0],  # only keep seconds list containing the coordinates
                filter(  # only keep valid coordinates
                    lambda elem_and_is_valid: elem_and_is_valid[1],
                    zip(
                        row["coordinates"], valid_osm_ids
                    ),  # combine coordinates and valid som_ids
                ),
            )
        )
        row["coordinates"] = valid_coordinates
        return row

    return df.apply(filter_func, axis=1)  # type: ignore


def print_time_and_speeds(filtered_df: pd.DataFrame, row: int = 7):
    """Used to verify the outlier from segment_10240935_linestring.json.

    Args:
        filtered_df (pd.DataFrame): dataframe, must have coordinates and speed cols
    """
    for (x, y, time), speed in zip(
        filtered_df.iloc[row]["coordinates"], filtered_df.iloc[7]["speeds"]
    ):
        print(time, speed, sep=", ")


def main():
    filename = "C:/Users/ax111/Documents/Personal documents/Coding/SW8/speed_limit_floating_car_data/archive/segment_10240935_linestring.json"
    df = create_df_from_json(filename)
    # university_boulevard_osm_id = 10240935
    # filtered_df = filter_segments(df, university_boulevard_osm_id)
    # calculate_distance_and_speed(df)
    avg, min, max = calculate_metrics(df)
    print(avg, min, max, sep=", ")  # keep


if __name__ == "__main__":
    main()
