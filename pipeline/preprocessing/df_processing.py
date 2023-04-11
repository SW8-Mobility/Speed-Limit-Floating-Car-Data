""" Used to a dataframe from geo json.
"""
import json
import pandas as pd  # type: ignore


def create_df_from_json(filename: str) -> pd.DataFrame:
    """Create dataframe for a linestring segment json file

    Returns:
        pd.DataFrame: dataframe with keys: ['id', 'length', 'end_date', 'start_date', 'osm_id', 'coordinates']
    """
    # url for segment 10240935:
    # https://fcd-share.civil.aau.dk/api/linestrings/?year=2014&osm_id=10240935&apikey=<API-KEY>
    with open(filename, "r") as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_records(  # convert json file to dataframe
        data["results"]["features"]
    )

    # unnest some of the nested values
    df["length"] = df["properties"].apply(lambda prop_dict: prop_dict["length"])
    df["end_date"] = df["properties"].apply(lambda prop_dict: prop_dict["end_date"])
    df["start_date"] = df["properties"].apply(lambda prop_dict: prop_dict["start_date"])
    df["osm_id"] = df["properties"].apply(lambda prop_dict: prop_dict["osm_id"])
    df["coordinates"] = df["geometry"].apply(
        lambda geometry_dict: geometry_dict["coordinates"]
    )

    df.drop(  # drop unused columns
        ["geometry", "properties", "type"], inplace=True, axis=1
    )  # drop unused columns

    df = df.infer_objects()  # infer types in dataframes

    return df


if __name__ == "__main__":
    pass
