import pandas as pd #type: ignore

from pipeline.preprocessing.ground_truth_processing.open_street.osm_geometry import annotate_df_with_geometry, \
    df_to_geo_json

@pytest.mark.parametrize(
    "input_dic, expected",
    [
        (
            {
                "osm_id":[2080],
                "cpr_vejnavn":["Holbækmotorvejen"],
                "hast_gaeldende_hast":[110],
                "osm_line_string":'{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}',
                "predicted_speed": [100]
            },
            '{"type": "FeatureCollection","features": [{"type": "Feature","geometry":{"type":"LineString","coordinates":'
            '[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]},'
            '"properties":{"osm_id":2080,"cpr_vejnavn":"Holbækmotorvejen","hast_gaeldende_hast":[110],"predicted_speed":[100]}}]'
        )
    ]
)
def test_df_to_geo_json(input_dic, expected):
    input_df = pd.DataFrame(data=input_dic)

    actual = df_to_geo_json(input_df)

    assert expected.equals(actual)


def test_annotate_df_with_geometry():
    # Arrange
    input_data = {
        "osm_id": [2080, 2081],
        "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen"],
        "hast_gaeldende_hast": [110, 110]}
    input_df = pd.DataFrame(data=input_data)

    input_dic = {}
    geo = "[[12.072963, 55.6345298],[12.0722614, 55.634588], [12.0711266, 55.6347107], [12.0705787, 55.6346358]]"
    input_dic[2080] = (geo, "insert_osm_name")
    input_dic[2081] = (geo, "insert_osm_name")

    expected = {
        "osm_id": [2080, 2081],
        "cpr_vejnavn": ["Holbækmotorvejen", "Holbækmotorvejen"],
        "hast_gaeldende_hast": [110, 110],
        "osm_line_string": [geo, geo]
    }
    expected_df = pd.DataFrame(data=expected)

    # Act
    annotate_df_with_geometry(input_df, input_dic) # mutates input_df

    # Assert
    assert input_df.equals(expected_df)

# {2080: ('{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}', 'Møllehusene')}
# {2081: ('{"type":"LineString","coordinates":[[12.0686169,55.6390613],[12.0685756,55.6392411],[12.0685724,55.6392531],[12.0685133,55.6394702],[12.0685055,55.639499],[12.0684818,55.6395862],[12.068465,55.6396479],[12.068336,55.6401221],[12.0680395,55.6412748],[12.0680057,55.6414063],[12.0674934,55.6431584],[12.0672676,55.6439174],[12.067135,55.6442801],[12.067133,55.6443301],[12.0671337,55.6443699],[12.0670927,55.6445252],[12.0670739,55.6445912],[12.0669844,55.6448857],[12.0669402,55.6450367],[12.0666259,55.6461149],[12.0665153,55.6465074],[12.0663765,55.6469954],[12.0663361,55.6473215],[12.0663341,55.6473655],[12.0663302,55.6475776],[12.0663288,55.6476466],[12.066336,55.6477368],[12.0663435,55.647831],[12.0663912,55.6484303],[12.0664006,55.6485489]]}', 'Møllehusvej')}
# {2080: ('{"type":"LineString","coordinates":[[12.072963,55.6345298],[12.0722614,55.634588],[12.0711266,55.6347107],[12.0705787,55.6346358]]}', 'Møllehusene'), 2081: ('{"type":"LineString","coordinates":[[12.0686169,55.6390613],[12.0685756,55.6392411],[12.0685724,55.6392531],[12.0685133,55.6394702],[12.0685055,55.639499],[12.0684818,55.6395862],[12.068465,55.6396479],[12.068336,55.6401221],[12.0680395,55.6412748],[12.0680057,55.6414063],[12.0674934,55.6431584],[12.0672676,55.6439174],[12.067135,55.6442801],[12.067133,55.6443301],[12.0671337,55.6443699],[12.0670927,55.6445252],[12.0670739,55.6445912],[12.0669844,55.6448857],[12.0669402,55.6450367],[12.0666259,55.6461149],[12.0665153,55.6465074],[12.0663765,55.6469954],[12.0663361,55.6473215],[12.0663341,55.6473655],[12.0663302,55.6475776],[12.0663288,55.6476466],[12.066336,55.6477368],[12.0663435,55.647831],[12.0663912,55.6484303],[12.0664006,55.6485489]]}', 'Møllehusvej')}