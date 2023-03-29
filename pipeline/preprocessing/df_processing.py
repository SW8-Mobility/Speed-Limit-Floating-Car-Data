""" Used to calculate speeds and distances for a dataframe.
"""

import json
from math import sqrt
import pandas as pd
from archive.analysis_metrics import calculate_metrics, filter_segments  # type: ignore

def shift_elems(l: list) -> list:
    """shift elemets of list, ie:
    [1,2,3,4] -> [None,1,2,3]
    will start with None.

    Args:
        l (list): input list of any type

    Returns:
        list: shifted list of same type
    """
    return [None] + l[:-1]


coordinate = tuple[float, float]  # type alias
def calc_utm_dist(utm1: coordinate, utm2: coordinate) -> float:
    """calculate the distance between two utm coordinates
    uses formula for euclidian distance, which is not 100% accurate?

    Args:
        utm1 (coordinate): a tuple of x and y coordinate
        utm2 (coordinate): a tuple of x and y coordinate

    Returns:
        float: distance between the two coordinates in meters
        (utm is already in meters, so no conversion)
    """
    x1, y1 = utm1
    x2, y2 = utm2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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


def ms_to_kmh(speeds: list[float]) -> list[float]:
    """convert list of speeds in meters per second to list of speeds with
    km per second.

    Args:
        speeds (list[float]): list of speeds

    Returns:
        list[float]: new list of speeds
    """
    return [speed * 3.6 for speed in speeds]

def calculate_distance_and_speed(df: pd.DataFrame):
    """Calculate the distance between each coordinate.
    Given that distance is recorded every second, the distance
    will also be the same as speed in meters per second.

    Args:
        df (pd.DataFrame): dataframe to calculate the distances on.
    """

    def calc_dist(row):
        coordinates = row["coordinates"]
        shifted_coordinates = row["shifted_coordinates"]
        shifted_coordinates.pop(
            0
        )  # Pop first element to avoid None, ok to mutate, since column is dropped later
        coordinates = coordinates[1:]  # also drop first, but not mutate

        distances = map(
            lambda cor_and_shif_cor: calc_utm_dist(
                (cor_and_shif_cor[0][0], cor_and_shif_cor[0][1]),
                (cor_and_shif_cor[1][0], cor_and_shif_cor[1][1]),
            ),
            zip(coordinates, shifted_coordinates),
        )
        return list(distances)

    df["shifted_coordinates"] = df["coordinates"].apply(shift_elems)
    df["distances"] = df.apply(calc_dist, axis=1)
    calculate_speeds(df)  # needs the shifted_coordinates column
    df.drop("shifted_coordinates", axis=1, inplace=True)


def calculate_speeds(df: pd.DataFrame) -> None:
    """Calculate the speeds for each coordinate for each trip

    Args:
        df (Pd.Dataframe): updated df with speeds
    """
    df["speeds"] = df["distances"].apply(ms_to_kmh)

    df[
        "time_difference"
    ] = df.apply(  # get time difference between each coordinate element
        lambda d: [
            c[2] - sc[2]
            for c, sc in zip(d["coordinates"][1:], d["shifted_coordinates"])
        ],
        axis=1,
    )

    # scale speed by time difference
    # # if two seconds have passed, then speed/2
    df["speeds"] = df.apply(
        lambda d: [
            speed / int(scale)
            for speed, scale in zip(d["speeds"], d["time_difference"])
        ],
        axis=1,
    )
    df.drop(["time_difference"], axis=1, inplace=True)



def main():
    filename = "segment_10240935_linestring.json"
    df = create_df_from_json(filename)
    university_boulevard_osm_id = 10240935
    filtered_df = filter_segments(df, university_boulevard_osm_id)
    calculate_distance_and_speed(filtered_df)
    avg, min, max = calculate_metrics(filtered_df)
    print(avg, min, max, sep=", ") # keep


if __name__ == "__main__":
    main()
