import json
import pandas as pd


def create_df_from_merged_osm_vejman(filename: str) -> pd.DataFrame:
    """
    Create a dataframe from merged between osm and vejman data.
    Args:
        filename: a file in geojson format

    Returns: dataframe with formated columns

    """

    with open(filename, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_records(  # Convert json file to dataframe
        data["features"]  # Ignore other headers
    )

    # Unnest some nested values
    property_list = [
        "osm_id",
        "CPR_VEJNAVN",
        "HAST_GENEREL_HAST",
        "KODE_HAST_GENEREL_HAST",
        "HAST_GAELDENDE_HAST",
        "VEJSTIKLASSE",
        "KODE_VEJSTIKLASSE",
        "VEJTYPESKILTET",
        "KODE_VEJTYPESKILTET",
        "HAST_SENEST_RETTET",
    ]
    unnest_df(df, "properties", property_list)

    geometry_list = ["coordinates"]
    unnest_df(df, "geometry", geometry_list)

    # drop unused columns
    df.drop(["type", "properties", "geometry"], inplace=True, axis=1)

    df = df.infer_objects()  # infer types in dataframes

    return df


def unnest_df(df: pd.DataFrame, nest_header: str, key_list: list[str]) -> None:
    """
    Unnest a given key from a column, if the format is like a dictionary. The column is added to the dataframe and the
    name will be in lower case.
    Args:
        df: a dataframe
        nest_header: the column where the nested values are
        key_list: a list of keys to look for
    """
    for key in key_list:
        df[key.lower()] = df[nest_header].apply(lambda dict: dict[key])


def main():
    filename = "C:/Users/freja/Desktop/Speed-Limit-Floating-Car-Data/tests/test_files/vejman_osm_merge_extract_data.json"
    df = create_df_from_merged_osm_vejman(filename)
    df.to_csv("from_fun.csv")


if __name__ == "__main__":
    main()
