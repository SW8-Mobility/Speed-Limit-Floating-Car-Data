import pytest
from pipeline.models.utils.scoring import find_closest_speed_limit

# SPEED_LIMITS = [15,30,40,50,60,70,80,90,100,110,120,130]

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
def test_none_if_empty(input, expected):
    assert find_closest_speed_limit(input) == expected