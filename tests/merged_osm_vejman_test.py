import os
import pandas as pd #type: ignore


def test_create_df_from_json():
    expected_data = {
        "osm_id": [1080, 2081, 2082],
        "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen", "Holbækmotorvejen"],

        "coordinates": [
            [ 12.072963, 55.6345298 ], [ 12.0722614, 55.634588 ], [ 12.0711266, 55.6347107 ], [ 12.0705787, 55.6346358 ]
            # Add the rest ...
        ],
    }
    expected_df = pd.DataFrame(data=expected_data)
    testfile_path = os.getcwd() + "/tests/test_files/merged_extract_v1.json"
    actual_df = (testfile_path)
    assert expected_df.equals(actual_df)