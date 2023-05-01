from pipeline.preprocessing.compute_features.feature import Feature

import pandas as pd  # type: ignore


def annotate_geojson_with_speedlimit(
    geojson: dict, df_with_speedlimit: pd.DataFrame
) -> None:
    """Annotates a geojson dict with predicted speedlimits from a dataframe.
    If there is no predicted value for a segment, the default will be 'na'.
    Args:
        geojson (dict): geojson as a dictionary
        df_with_speedlimit (pd.DataFrame): dataframe with speedlimits
    """
    # index using osm_id - this makes look ups faster
    attribute_name = "speed limit"
    df_with_speedlimit["index"] = df_with_speedlimit[Feature.OSM_ID.value]
    df_with_speedlimit = df_with_speedlimit.set_index("index")

    for segment in geojson["features"]:
        osm_id = segment["properties"]["osm_id"]
        try:
            segment["properties"][attribute_name] = df_with_speedlimit.loc[osm_id][
                Feature.SPEED_LIMIT_PREDICTED.value
            ]
        except:  # no predicted value for osm_id
            segment["properties"][attribute_name] = "na"
