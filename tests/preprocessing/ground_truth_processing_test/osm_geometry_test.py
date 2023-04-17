import pandas as pd  # type: ignore
import pytest
from pipeline.preprocessing.compute_features.feature import Feature
from pipeline.preprocessing.ground_truth_processing.open_street.osm_geometry import (
    annotate_df_with_osm_data,
    annotate_geojson_with_speedlimit,
    df_to_geo_json,
)


# @pytest.mark.parametrize(
#     "input_dic, expected",
#     [
#         (
#             {
#                 "osm_id": 2080,
#                 "cpr_vejnavn": "Holbækmotorvejen",
#                 "hast_gaeldende_hast": 110,
#                 "predicted_speed": 100,
#                 "osm_line_string": '{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}',
#                 "osm_name": "Møllehusene",
#             },
#             '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"LineString","coordinates":'
#             "[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]},"
#             '"properties":{"osm_id":2080,"cpr_vejnavn":"Holbækmotorvejen","hast_gaeldende_hast":110,"predicted_speed":100,'
#             '"osm_name":"Møllehusene"}}]}',
#         ),
#         (
#             {
#                 "osm_id": [2080, 2081],
#                 "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen"],
#                 "hast_gaeldende_hast": [110, 110],
#                 "predicted_speed": [100, 110],
#                 "osm_line_string": [
#                     '{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}',
#                     '{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}',
#                 ],
#                 "osm_name": ["Møllehusene", "Møllehusvej"],
#             },
#             '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"LineString","coordinates":'
#             "[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]},"
#             '"properties":{"osm_id":2080,"cpr_vejnavn":"Holbækmotorvejen","hast_gaeldende_hast":110,"predicted_speed":100,'
#             '"osm_name":"Møllehusene"}},'
#             '{"type":"Feature","geometry":{"type":"LineString","coordinates":'
#             "[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]},"
#             '"properties":{"osm_id":2080,"cpr_vejnavn":"Holbækmotorvejen","hast_gaeldende_hast":110,"predicted_speed":110,'
#             '"osm_name":"Møllehusvej"}}]}',
#         ),
#     ],
# )
# def test_df_to_geo_json(input_dic, expected):
#     input_df = pd.DataFrame(data=input_dic)

#     actual = df_to_geo_json(input_df)

#     with open("test_res.txt", "w", encoding="utf-8") as f:
#         f.write(f"\n\nactual: {actual} \n\nexpected: {expected}")

#     test1 = expected.encode("utf-8")
#     test2 = actual.encode("utf-8")
#     assert test1 == test2


# @pytest.mark.parametrize(
#     "input_data, input_dict, expected_data",
#     [
#         (
#             {
#                 "osm_id": [2080, 2081],
#                 "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen"],
#                 "hast_gaeldende_hast": [110, 110],
#                 "predicted_speed": [42, 42],
#             },
#             {
#                 2080: (
#                     "[[12.072963, 55.6345298],[12.0722614, 55.634588]",
#                     "some_osm_name",
#                 ),
#                 2081: (
#                     "[[12.072963, 55.6345298],[12.0722614, 55.634588]",
#                     "some_osm_name2",
#                 ),
#             },
#             {
#                 "osm_id": [2080, 2081],
#                 "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen"],
#                 "hast_gaeldende_hast": [110, 110],
#                 "predicted_speed": [42, 42],
#                 "osm_line_string": [
#                     "[[12.072963, 55.6345298],[12.0722614, 55.634588]",
#                     "[[12.072963, 55.6345298],[12.0722614, 55.634588]",
#                 ],
#                 "osm_name": ["some_osm_name", "some_osm_name2"],
#             },
#         )
#     ],
# )
# def test_annotate_df_with_osm_data(input_data, input_dict, expected_data):
#     # Arrange
#     actual = pd.DataFrame(data=input_data)
#     expected_df = pd.DataFrame(data=expected_data)

#     # Act
#     annotate_df_with_osm_data(actual, input_dict)  # mutates 'actual' df

#     # Assert
#     assert actual.equals(expected_df)


@pytest.mark.parametrize(
    "df_data, geojson_dict, expected_dict",
    [
        (
            {
                "osm_id": [2080, 2081],
                Feature.SPEED_LIMIT.value: [110, 130],
            },
            {
                "features": [
                    {
                        'properties': {'osm_id': 2080}
                    },
                    {
                        'properties': {'osm_id': 2081}
                    },
                ]
            },
            {
                "features": [
                    {
                        'properties': {'osm_id': 2080, Feature.SPEED_LIMIT.value: 110}
                    },
                    {
                        'properties': {'osm_id': 2081, Feature.SPEED_LIMIT.value: 130}
                    },
                ]
            }
        )
    ],
)
def test_annotate_geojson_with_speedlimit(df_data, geojson_dict, expected_dict):
    # Arrange
    df = pd.DataFrame(data=df_data)

    # Act
    annotate_geojson_with_speedlimit(geojson_dict, df)

    # Assert
    assert geojson_dict == expected_dict
