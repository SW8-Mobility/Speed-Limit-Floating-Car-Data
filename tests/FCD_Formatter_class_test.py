import pandas as pd  # type: ignore
import pytest

from pipeline.preprocessing.formatting.FCD_Formatter import (
    FCD_Formatter,
    _create_segment_to_coordinate_df,
    _map_segments_to_coordinates,
    _clean_df,
)


@pytest.mark.parametrize(
    "segments, coordinates, expected",
    [
        (
            [1, 1, 1, 2, 2, 2, 3, 3, 3, None, None],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [(1, [1, 2, 3]), (2, [4, 5, 6]), (3, [7, 8, 9]), (None, [10, 11])],
        ),
        ([1], [1], [(1, [1])]),
        ([], [], []),
    ],
)
def test_map_segments_to_coordinates(segments, coordinates, expected):
    assert _map_segments_to_coordinates(segments, coordinates) == expected


def test_create_segment_to_coordinate_df_one_segment():
    expected_data = {
        "osm_id": [1234],
        "coordinates": [
            [
                [
                    [100.0, 100.0, 1000000000.0],
                    [100.0, 110.0, 1000000001.0],
                    [100.0, 130.0, 1000000002.0],
                ]
            ]
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "osm_id": [[1234, 1234, 1234]],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ]
        ],
    }
    df = pd.DataFrame(data=actual_data)

    actual_df = _create_segment_to_coordinate_df(df)

    assert expected_df.equals(actual_df)


def test_create_segment_to_coordinate_df_one_segment_with_none():
    expected_data = {
        "osm_id": [1234],
        "coordinates": [
            [
                [
                    [100.0, 100.0, 1000000000.0],
                    [100.0, 110.0, 1000000001.0],
                    [100.0, 130.0, 1000000002.0],
                ]
            ]
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "osm_id": [[1234, 1234, 1234, None]],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
                [100.0, 140.0, 1000000003.0],
            ]
        ],
    }

    df = pd.DataFrame(data=actual_data)
    actual_df = _create_segment_to_coordinate_df(df)

    assert expected_df.equals(actual_df)


def test_create_segment_to_coordinate_df_multiple_segments():
    expected_data = {
        "osm_id": [1111, 2222, 3333],
        "coordinates": [
            [
                [
                    [100.0, 100.0, 1000000000.0],
                    [100.0, 110.0, 1000000001.0],
                    [100.0, 130.0, 1000000002.0],
                ]
            ],
            [
                [
                    [200.0, 100.0, 1000000000.0],
                    [200.0, 110.0, 1000000001.0],
                    [200.0, 130.0, 1000000002.0],
                ]
            ],
            [
                [
                    [300.0, 100.0, 1000000000.0],
                    [300.0, 110.0, 1000000001.0],
                    [300.0, 130.0, 1000000002.0],
                ]
            ],
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "osm_id": [
            [1111, 1111, 1111],
            [2222, 2222, 2222],
            [3333, 3333, 3333],
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
            [
                [200.0, 100.0, 1000000000.0],
                [200.0, 110.0, 1000000001.0],
                [200.0, 130.0, 1000000002.0],
            ],
            [
                [300.0, 100.0, 1000000000.0],
                [300.0, 110.0, 1000000001.0],
                [300.0, 130.0, 1000000002.0],
            ],
        ],
    }

    df = pd.DataFrame(data=actual_data)
    actual_df = _create_segment_to_coordinate_df(df)

    assert expected_df.equals(actual_df)


@pytest.mark.parametrize(
    "osm_id, expected_osm, expected_coordinates",
    [
        (
            [1111, 1111, 1111, None],
            [1111, 1111, 1111],
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
        ),
        (
            [1111, None, 1111, 1111],
            [1111, 1111, 1111],
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 130.0, 1000000002.0],
                [100.0, 140.0, 1000000003.0],
            ],
        ),
        (
            [1111, None, None, 1111],
            [1111, 1111],
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 140.0, 1000000003.0],
            ],
        ),
        ([None, None, None, None], [], []),
    ],
)
def test_clean_df(osm_id, expected_osm, expected_coordinates):
    expected_data = {
        "osm_id": [expected_osm],
        "coordinates": [expected_coordinates],
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "osm_id": [
            osm_id,
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
                [100.0, 140.0, 1000000003.0],
            ]
        ],
    }
    actual_df = pd.DataFrame(data=actual_data)
    _clean_df(actual_df)

    assert expected_df.equals(actual_df)
