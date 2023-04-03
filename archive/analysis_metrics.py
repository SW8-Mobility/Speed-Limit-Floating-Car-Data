""" Some functions used to inspect and anlysize our data.
"""

from functools import reduce
import json
from math import sqrt
from itertools import tee as copy_iterable
from statistics import median, mean
import sys
from typing import Iterator
import pandas as pd  # type: ignore


def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate aggregate min, max, and avg for dataframe

    Args:
        df (pd.DataFrame): Dataframe to calculate on

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


def verify_solution(l: tuple[Iterator[bool], Iterator[bool]]) -> None:
    """Auxiliary fuction used in filter_segments, to verify the solution works.
    I assume that trips do not loop back and go through the same
    segment more than once. This function is a sanitity check for this.

    Args:
        l (list[bool]): list with boolean mask of all coordinates that are in
        the segment of interest.
    """
    safe_cmp = lambda b, acc: True if len(acc) == 0 else acc[-1] != b
    reduced: Iterator[
        bool
    ] = reduce(  # convert [False, False, True, True, True, False] -> [False, True, False]
        lambda acc, bool: acc + [bool] if safe_cmp(bool, acc) else acc, l, []  # type: ignore
    )

    # if there are more than one true in list
    # then the car has looped and break assumption
    if len(list(filter(lambda e: e is True, reduced))) > 1:  # type: ignore
        print("Bad assumption...")
        sys.exit()


def filter_segments(df: pd.DataFrame, osm_id: int) -> pd.DataFrame:
    """Discard all rows from coordinates that do not correspond to the given osm_id.

    Args:
        df (pd.DataFrame): geodata dataframe
        osm_id (int): segment id

    Returns:
        pd.DataFrame: dataframe only with coordinates corresponding to given osm_id
    """

    def filter_func(row):
        """read above doc string"""
        valid_osm_ids = map(  # id mask corresponding to which coordinates to keep
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
    The speed calculated was initially wrong. But now fixed.

    Args:
        filtered_df (pd.DataFrame): dataframe
    """
    for (x, y, time), speed in zip(
        filtered_df.iloc[row]["coordinates"], filtered_df.iloc[7]["speeds"]
    ):
        print(time, speed, sep=", ")


if __name__ == "__main__":
    pass
