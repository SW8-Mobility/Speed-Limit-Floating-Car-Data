import pytest

from pipeline.preprocessing.compute_features.vcr_calculator import map_vcr, vcr, TooFewElementsException


@pytest.mark.parametrize(
    "test_input, expected_vcrs",
    [
        (
            f"{map_vcr([1,2,3,4,5,6,7])}",
            [
                0.5,
                0.3333333333333333,
                0.25,
                0.2,
                0.16666666666666666,
                0.14285714285714285,
            ],
        ),
        (f"{map_vcr([3, 2])}", [-0.5]),
    ],
)
def test_map_vcr(test_input, expected_vcrs: list[float]) -> None:
    assert eval(test_input) == expected_vcrs


@pytest.mark.parametrize(
    "test_input, expected_vcr",
    [(f"{vcr(55.78, 45.0)}", -0.23955555555555558), (f"{vcr(40, 50)}", 0.2)],
)
def test_vcr(test_input, expected_vcr) -> None:
    assert eval(test_input) == expected_vcr


@pytest.mark.parametrize(
    "test_input",
    [f"{[]}", f"{[100]}", f"{[-100]}", f"{[0]}", f"{[50]}"],
)
def test_too_few_elements_exception(test_input):
    with pytest.raises(TooFewElementsException):
        map_vcr(test_input)
