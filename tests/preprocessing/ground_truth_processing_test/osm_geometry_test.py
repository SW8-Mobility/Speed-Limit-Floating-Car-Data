import pandas as pd  # type: ignore
import pytest
from pipeline.preprocessing.compute_features.feature import Feature
from pipeline.preprocessing.ground_truth_processing.open_street.osm_geometry import (
    annotate_geojson_with_speedlimit,
)

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
                    {"properties": {"osm_id": 2080}},
                    {"properties": {"osm_id": 2081}},
                ]
            },
            {
                "features": [
                    {"properties": {"osm_id": 2080, Feature.SPEED_LIMIT.value: 110}},
                    {"properties": {"osm_id": 2081, Feature.SPEED_LIMIT.value: 130}},
                ]
            },
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


@pytest.mark.parametrize(
    "df_data, geojson_dict, expected_dict",
    [
        (
            {
                "osm_id": [2080],
                Feature.SPEED_LIMIT.value: [110],
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
                        'properties': {'osm_id': 2081, Feature.SPEED_LIMIT.value: 'na'}
                    },
                ]
            }
        )
    ],
)
def test_annotate_geojson_with_speedlimit_missing_entry(df_data, geojson_dict, expected_dict):
    # Arrange
    df = pd.DataFrame(data=df_data)

    # Act
    annotate_geojson_with_speedlimit(geojson_dict, df)

    # Assert
    assert geojson_dict == expected_dict

