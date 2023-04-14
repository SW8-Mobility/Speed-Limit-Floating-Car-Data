import pytest
from pipeline.models.utils.scoring import (
    find_closest_speed_limit,
    mean_absolute_percentage_error,
)

@pytest.mark.parametrize(
    "input, expected",
    [
        (-1000, 15),
        (0, 15),
        (1, 15),
        (9, 15),
        (20, 15),
        (25, 30),
        (31, 30),
        (41, 40),
        (45, 40),
        (50, 50),
        (51, 50),
        (55, 50),
        (56, 60),
        (101, 100),
        (125, 120),
        (40000, 130),
    ],
)
def test_find_closest_speed_limit(input, expected):
    assert find_closest_speed_limit(input) == expected


@pytest.mark.parametrize(
    "ground_truth, predictions, expected",
    [
        ([10, 20, 30], [10, 20, 30], 0),
        ([10, 20, 30], [11, 19, 29], 6.1111),
        ([10, 20, 30], [5, 15, 25], 30.5555),
        ([-10, -20, -30], [-8, -22, -32], 12.2222),
    ],
)
def test_mean_absolute_percentage_error(ground_truth, predictions, expected):
    assert (
        pytest.approx(mean_absolute_percentage_error(ground_truth, predictions), 0.0001)
        == expected
    )
