import pytest

from pipeline.preprocessing.compute_features.vcr_calculator import (
    map_vcr,
    vcr,
    TooFewElementsException,
)


@pytest.mark.parametrize(
    "test_input, expected_vcrs",
    [
        (
            [1, 2, 3, 4, 5, 6, 7],
            [
                0.5,
                0.3333333333333333,
                0.25,
                0.2,
                0.16666666666666666,
                0.14285714285714285,
            ],
        ),
        ([3, 2], [-0.5]),
        ([20, 20, 0, 20, 20], [0, 1.0, 0]),
    ],
)
def test_map_vcr(test_input, expected_vcrs: list[float]) -> None:
    assert map_vcr(test_input) == expected_vcrs


@pytest.mark.parametrize(
    "v1, v2, expected_vcr",
    [(55.78, 45.0, -0.23955555555555558), (40, 50, 0.2)],
)
def test_vcr(v1, v2, expected_vcr) -> None:
    assert vcr(v1, v2) == expected_vcr


@pytest.mark.parametrize(
    "test_input",
    [[], [100], [-100], [0], [50]],
)
def test_too_few_elements_exception(test_input):
    with pytest.raises(TooFewElementsException):
        map_vcr(test_input)
