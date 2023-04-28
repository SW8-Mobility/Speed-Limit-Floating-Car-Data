import pytest
import json
import os
import pandas as pd  # type: ignore
from pipeline.preprocessing.formatting.FCD_Formatter import _remove_fcd_request_wrapper


def test_create_df_from_json():
    expected_data = {
        "id": "test_id",
        "length": 1234,
        "end_date": "test_date",
        "start_date": "test_date",
        "osm_id": [[123456789, 123456789, 123456789]],
        "coordinates": [
            [
                [100.0, 100.0, 1000000000.0],
                [100.0, 110.0, 1000000001.0],
                [100.0, 130.0, 1000000002.0],
            ]
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)
    testfile_path = os.getcwd() + "/tests/test_files/geo_json_trip_data.json"
    with open(testfile_path, 'r') as testfile:
        data = json.load(testfile)
        actual_df = _remove_fcd_request_wrapper(data)
        assert expected_df.equals(actual_df)
