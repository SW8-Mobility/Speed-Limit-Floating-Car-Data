from geo_json_metrics.gps_metrics import calc_utm_dist


def test_calc_utm_dist() -> None:
    # Arrange
    coordinate = tuple[float, float]  # type alias
    cord1: coordinate = (2, 3)
    cord2: coordinate = (5, 3)
    expected_utm_dist: float = 3

    # Act
    actual_utm_dist: float = calc_utm_dist(cord1, cord2)

    # Assert
    assert expected_utm_dist == actual_utm_dist
