import sys
from os.path import dirname, realpath

current_dir = dirname(realpath(__file__))
parentdir = dirname(dirname(dirname(current_dir)))
sys.path.append(parentdir)

from cmath import sqrt
import pytest
import pipeline.preprocessing.compute_features.calculate_speeds_distances as csd


@pytest.mark.parametrize(
    "utm1, utm2, expected",
    [
        (
            (3, 2),
            (4, 1),
            sqrt(2),
        ),
        ((2, 2), (2, 4), 2),
    ],
)
def test_calc_utm_dist(utm1, utm2, expected):
    assert csd.calc_utm_dist(utm1, utm2) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
            [10.0, 20.0],
        ),
        (
            [
                [100.0, 100.0, 1000000000.0],
            ],
            [],
        ),
        (
            [
                [100.0, 100.0, 1000000000.0],
                [150.0, 100.0, 1000000001.0],
                [200.0, 100.0, 1000000002.0],
            ],
            [50.0, 50.0],
        ),
    ],
)
def test_calculate_distances(input, expected):
    assert csd.calculate_distances(input) == expected


@pytest.mark.parametrize(
    "input, distances, expected",
    [
        (
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
            [10.0, 20.0],
            [36.0, 72.0],
        ),
        (
            [
                [100.0, 100.0, 1000000000.0],
            ],
            [],
            [],
        ),
        (
            [
                [100.0, 100.0, 1000000000.0],
                [150.0, 100.0, 1000000001.0],
                [200.0, 100.0, 1000000002.0],
            ],
            [50.0, 50.0],
            [180.0, 180.0],
        ),
    ],
)
def test_calculate_speeds(input, distances, expected):
    assert csd.calculate_speeds(input, distances) == expected
