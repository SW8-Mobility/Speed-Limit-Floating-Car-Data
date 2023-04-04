import json
import pandas as pd  # type: ignore


def create_df_from_merged_osm_vejman(filename: str) -> pd.DataFrame:
    with open(filename, "r") as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_records(  # Convert json file to dataframe
        data["features"]  # Ignore other headers
    )

    # Unnest some of the nested values - TODO: this could easily be a function
    df["osm_id"] = df["properties"].apply(lambda prop_dict: prop_dict["osm_id"])
    df["vm_vejnavn"] = df["properties"].apply(
        lambda prop_dict: prop_dict["CPR_VEJNAVN"]
    )
    df["hast_generel_hast"] = df["properties"].apply(
        lambda prop_dict: prop_dict["HAST_GENEREL_HAST"]
    )
    df["kode_hast_generel_hast"] = df["properties"].apply(
        lambda prop_dict: prop_dict["KODE_HAST_GENEREL_HAST"]
    )
    df["hast_gaeldende_hast"] = df["properties"].apply(
        lambda prop_dict: prop_dict["HAST_GAELDENDE_HAST"]
    )
    df["vejstiklasse"] = df["properties"].apply(
        lambda prop_dict: prop_dict["VEJSTIKLASSE"]
    )
    df["kode_vejstiklasse"] = df["properties"].apply(
        lambda prop_dict: prop_dict["KODE_VEJSTIKLASSE"]
    )
    df["vejtypeskiltet"] = df["properties"].apply(
        lambda prop_dict: prop_dict["VEJTYPESKILTET"]
    )
    df["kode_vejtypeskiltet"] = df["properties"].apply(
        lambda prop_dict: prop_dict["KODE_VEJTYPESKILTET"]
    )
    df["hast_senest_rettet"] = df["properties"].apply(
        lambda prop_dict: prop_dict["HAST_SENEST_RETTET"]
    )

    df["coordinates"] = df["geometry"].apply(lambda prop_dict: prop_dict["coordinates"])

    df.drop(  # drop unused columns
        ["type", "properties", "geometry"], inplace=True, axis=1
    )

    df = df.infer_objects()  # infer types in dataframes

    return df
