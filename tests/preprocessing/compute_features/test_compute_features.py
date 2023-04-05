from statistics import mean, median

import pandas as pd  # type: ignore
import pytest
import pipeline.preprocessing.compute_features.compute_features as cf
from pipeline.preprocessing.compute_features.feature import Feature


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

# TODO: check where none/empty lists in coloumns should be handled


def test_per_trip_speed_computation(func, speed_column, expected):
    df = pd.DataFrame(data={Feature.SPEEDS.value: speed_column})
    assert cf.per_trip_speed_computation(func, df) == expected
