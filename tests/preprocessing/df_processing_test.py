import pytest
import os
import pandas as pd  # type: ignore
from pipeline.preprocessing.formatting.FCD_Formatter import FCD_Formatter


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
    actual_df = FCD_Formatter.from_json_file(testfile_path)


    
    print(actual_df)
    print(expected_data)

    assert expected_df.equals(actual_df)
