from geo_json_metrics.vcr_calculator import map_vcr, vcr


def test_map_vcr() -> None:
    # Arrange
    testList: list[float] = [1, 2, 3, 4, 5, 6, 7]
    expected_vcrs: list[float] = [0.5, 0.33333333333, 0.25, 0.2, 0.16666666666, 14285714285]

    # Act
    actual_vcrs: list[float] = map_vcr(testList)

    # Assert
    assert actual_vcrs == expected_vcrs
    assert len(actual_vcrs) == len(testList)-1


def test_vcr() -> None:
    # Arrange
    v1: float = 55.78
    v2: float = 45.0
    expected_vcr: float = -0.23955555555555558

    # Act
    actual_vcr: float = vcr(v1, v2)

    # Assert
    assert actual_vcr == expected_vcr
    assert actual_vcr < 0
