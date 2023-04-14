import pytest
import os
import pandas as pd  # type: ignore
from pipeline.preprocessing.process_merged_geojson import (
    create_df_from_merged_osm_vejman,
)


def test_process_merged_osm_from_json():
    expected_data = {
        "osm_id": [2080, 2081, 2082],
        "osm_name": ["Møllehusene", "Møllehusvej", "Byvolden"],
        "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen", "Holbækmotorvejen"],
        "hast_generel_hast": ["130 - Motorvej", "130 - Motorvej", "130 - Motorvej"],
        "kode_hast_generel_hast": ["130", "130", "130"],
        "hast_gaeldende_hast": [110, 110, 110],
        "vejstiklasse": [
            "Trafikvej, gennemfart land",
            "Trafikvej, gennemfart land",
            "Trafikvej, gennemfart land",
        ],
        "kode_vejstiklasse": ["1", "1", "1"],
        "vejtypeskiltet": ["Motorvej", "Motorvej", "Motorvej"],
        "kode_vejtypeskiltet": ["1", "1", "1"],
        "hast_senest_rettet": [
            "2006-08-04T05:45:54",
            "2011-12-03T06:27:52",
            "2006-08-04T05:45:54",
        ],
        "coordinates": [
            [
                [12.072963, 55.6345298],
                [12.0722614, 55.634588],
                [12.0711266, 55.6347107],
                [12.0705787, 55.6346358],
            ],
            [
                [12.0686169, 55.6390613],
                [12.0685756, 55.6392411],
                [12.0685724, 55.6392531],
                [12.0685133, 55.6394702],
                [12.0685055, 55.639499],
                [12.0684818, 55.6395862],
                [12.068465, 55.6396479],
                [12.068336, 55.6401221],
                [12.0680395, 55.6412748],
                [12.0680057, 55.6414063],
                [12.0674934, 55.6431584],
                [12.0672676, 55.6439174],
                [12.067135, 55.6442801],
                [12.067133, 55.6443301],
                [12.0671337, 55.6443699],
                [12.0670927, 55.6445252],
                [12.0670739, 55.6445912],
                [12.0669844, 55.6448857],
                [12.0669402, 55.6450367],
                [12.0666259, 55.6461149],
                [12.0665153, 55.6465074],
                [12.0663765, 55.6469954],
                [12.0663361, 55.6473215],
                [12.0663341, 55.6473655],
                [12.0663302, 55.6475776],
                [12.0663288, 55.6476466],
                [12.066336, 55.6477368],
                [12.0663435, 55.647831],
                [12.0663912, 55.6484303],
                [12.0664006, 55.6485489],
            ],
            [
                [12.0741272, 55.6437429],
                [12.0738026, 55.6433187],
                [12.0737126, 55.6432011],
                [12.0736049, 55.6430591],
                [12.073289, 55.6426658],
                [12.072902, 55.6421562],
                [12.0722318, 55.6412806],
                [12.0718806, 55.6408221],
                [12.0718337, 55.640695],
            ],
        ],
    }
    expected_df = pd.DataFrame(data=expected_data).astype(
        {"hast_gaeldende_hast": "float64"}
    )

    testfile_path = os.getcwd() + "/tests/test_files/vejman_osm_merge_extract_data.json"
    # TODO: file path works when running "pytest .", but will fail when running from file...
    actual_df = create_df_from_merged_osm_vejman(testfile_path)

    assert expected_df.equals(actual_df)
