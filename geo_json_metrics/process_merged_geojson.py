import json
import pandas as pd

def create_df_from_json(filename: str) -> pd.DataFrame:

    with open(filename, "r") as json_file:
        data = json.load(json_file)

    df = pd.DataFrame.from_records(  # Convert json file to dataframe
        data["features"] # Ignore other headers
    )

    # Unnest some of the nested values
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

    print(df)

    return df

def main():
    filename = "C:/Users/freja/Desktop/Speed-Limit-Floating-Car-Data/tests/test_files/merged_extract_v1.json"
    df = create_df_from_json(filename)

if __name__ == "__main__":
    main()