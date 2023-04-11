from statistics import mean, median

import pandas as pd  # type: ignore
import pytest
import pipeline.preprocessing.compute_features.compute_features as cf
from pipeline.preprocessing.compute_features.feature import Feature

def assert_lists_almost_equal(list1: list[float], list2: list[float], tolerance: float):
    """To account for impreciseness in float computations, this function will
    check if two lists of floats are equal, with some tolerence for difference.

    Args:
        list1 (list[float]): actual
        list2 (list[float]): expected
        tolerance (float): tolerance for impreciseness
    """
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2):
         assert pytest.approx(a, tolerance) == b

@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [1, 2, 3],
            42,
        ),
        ([1, None, 3], 42),
        ([], None),
    ],
)
def test_none_if_empty(input, expected):
    func = lambda x: 42  # Function that always returns 42
    assert cf.none_if_empty(func, input) == expected


@pytest.mark.parametrize(
    "func, speed_column, expected",
    [
        (min, [[23, 42, 82], [47, 58, 60]], [23, 47]),
        (max, [[0, 47, 88], [89, 200, 67]], [88, 200]),
        (mean, [[1, 2], [4, 4]], [1.5, 4]),
        (median, [[1, 2, 3, 4, 5]], [3]),
    ],
)

def test_per_trip_speed_computation(func, speed_column, expected):
    df = pd.DataFrame(data={Feature.SPEEDS.value: speed_column})
    assert cf.per_trip_speed_computation(func, df) == expected

@pytest.mark.parametrize(
    "input_speeds, k, expected",
    [
        ([], 3, []),
        ([1], 3, []),
        ([1,2], 3, []),
        ([1,2,3], 3, [2]),
        ([1,2,3,4,5,6], 3, [2,3,4,5]),
        ([2,4,6,8,12,14,16,18,20], 3, [4,6,8.667,11.333,14,16,18]),
        ([80,85,100,60,40,20], 5, [73,61]),
        ([100,90,80,50,20,40,60,70,100], 4, [80,60,47.5,42.5,47.5,67.5]),
        ([10.1,40.5,45.5,63.8,91.3,100.9], 3, [32.033,49.933,66.867,85.333]),
    ],
)
def test_k_rolling_avg(input_speeds, k, expected):
    actual = cf.k_rolling_avg(input_speeds, k)
    assert_lists_almost_equal(actual, expected, 0.0001)
