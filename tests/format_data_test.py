import pytest
from geo_json_metrics.format_data import map_segments_to_coordinates


@pytest.mark.parametrize(
    "segments, coordinates, expected",
    [
        (
            [1, 1, 1, 2, 2, 2, 3, 3, 3, None, None],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [(1, [1, 2, 3]), (2, [4, 5, 6]), (3, [7, 8, 9]), (None, [10, 11])]
        ),
        (
            [1], [1], [(1, [1])]
        ),
        (
            [], [], []
        )
    ],
)
def test_map_segments_to_coordinates(segments, coordinates, expected):
    assert map_segments_to_coordinates(segments, coordinates) == expected