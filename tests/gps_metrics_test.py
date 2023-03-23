from cmath import sqrt
import os
from typing import Union
import numpy as np
import pandas as pd
import pytest
from geo_json_metrics.gps_metrics import shift_elems, calc_utm_dist, create_df_from_json, calculate_distance_and_speed


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            [1, 2, 3, 4],
            [None, 1, 2, 3],
        ),
    ],
)
def test_shift_elems(test_input: list, expected: list) -> None:
    assert shift_elems(test_input) == expected


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
    assert calc_utm_dist(utm1, utm2) == expected

def test_create_df_from_json():
    expected_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ]
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)
    testfile_path = os.getcwd() + "\\tests\\test_files\\test_json.txt"
    actual_df = create_df_from_json(testfile_path)
    assert expected_df.equals(actual_df)

def test_calculate_distance_and_speed():
    expected_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ]
        ],
        "distances": [
            [
                10.0, 20.0
            ]
        ],
        "speeds": [
            [
                36.0, 72.0
            ]
        ]
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ]
        ]
    }
    actual_df = pd.DataFrame(data=actual_data)
    calculate_distance_and_speed(actual_df)
    assert actual_df.speeds.equals(expected_df.speeds)

def test_calculate_distance_and_speed2():
    expected_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
            ]
        ],
        "distances": [
            []
        ],
        "speeds": [
            []
        ]
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
            ]
        ]
    }
    actual_df = pd.DataFrame(data=actual_data)
    calculate_distance_and_speed(actual_df)
    assert actual_df.speeds.equals(expected_df.speeds)


def test_calculate_distance_and_speed3():
    expected_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789],
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
            [
                [100.0, 100.0, 1000000000.0],
                [150.0, 100.0, 1000000001.0],
                [200.0, 100.0, 1000000002.0],
            ]
        ],
        "distances": [
            [10.0, 20.0],
            [50.0, 50.0]
        ],
        "speeds": [
            [36.0, 72.0],
            [180.0, 180.0]
        ]
    }
    expected_df = pd.DataFrame(data=expected_data)

    actual_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [
            [123456789, 123456789, 123456789],
            [123456789, 123456789, 123456789]
        ],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ],
            [
                [100.0, 100.0, 1000000000.0],
                [150.0, 100.0, 1000000001.0],
                [200.0, 100.0, 1000000002.0],
            ]
        ]
    }
    actual_df = pd.DataFrame(data=actual_data)
    calculate_distance_and_speed(actual_df)
    assert actual_df.speeds.equals(expected_df.speeds)