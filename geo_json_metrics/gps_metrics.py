from functools import reduce
import json
from math import sqrt
import pandas as pd  # type: ignore
from itertools import tee as copy_iterable
from statistics import median, mean

"""Warning: there is an assumption that for all trips, the distance is updated every second
this is NOT the case, sometimes there are a few "jumps" in time ie. more than 1 second between 
each recorded distance.
Should be fixed, if script needs to be used for more data...
"""

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
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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


def filter_segments(df: pd.DataFrame, osm_id: int) -> pd.DataFrame:
    """Discard all rows from coordinates that do not correspond to the given osm_id.

    Args:
        df (pd.DataFrame): geodata dataframe
        osm_id (int): segment id

    Returns:
        pd.DataFrame: dataframe only with coordinates corresponding to given osm_id
    """

    def verify_solution(l: list[bool]):
        """I assume that trips do not loop back and go through the same
        segment more than once. This function is a sanitity check for this.

        Args:
            l (list[bool]): list with boolean mask of all coordinates that are in
            the segment of interest.
        """
        safe_cmp = lambda b, acc: True if len(acc) == 0 else acc[-1] != b
        reduced: list[bool] = reduce(  # convert [False, False, True, True, True, False] -> [False, True, False]
            lambda acc, bool: acc + [bool] if safe_cmp(bool, acc) else acc, l, object()
        )

        # if there are more than one true in list
        # then the car has looped and break assumption
        if len(list(filter(lambda e: e == True, reduced))) > 1:
            print("Bad assumption...")
            exit()

    def filter_func(row):
        valid_osm_ids = map(  # id mask corresponding to which coordinates to keep
            lambda elem: elem == osm_id, row["osm_id"]
        )
        verify_solution(copy_iterable(valid_osm_ids))

        valid_coordinates = list(
            map(
                lambda tup: tup[0],  # only keep seconds list containing the coordinates
                filter(  # only keep valid coordinates
                    lambda elem_and_is_valid: elem_and_is_valid[1],
                    zip(
                        row["coordinates"], valid_osm_ids
                    ),  # combine coordinates and valid som_ids
                ),
            )
        )
        row["coordinates"] = valid_coordinates
        return row

    return df.apply(filter_func, axis=1)  # type: ignore


def calculate_distance(df: pd.DataFrame):
    """Calculate the distance between each coordinate.
    Given that distance is recorded every second, the distance
    will also be the same as speed in meters per second.

    Args:
        df (pd.DataFrame): dataframe to calculate the distances on.
    """

    def shift_elems(l: list):
        l.pop(-1)
        return [None] + l

    df["shifted_coordinates"] = df["coordinates"].apply(
        lambda coordinates: shift_elems(coordinates)
    )

    def calc_dist(row):
        coordinates = row["coordinates"]
        shifted_coordinates = row["shifted_coordinates"]
        shifted_coordinates.pop(0)  # Pop first element to avoid None
        coordinates.pop(0)  # Pop first element to have equal size

        distances = map(
            lambda cor_and_shif_cor: calc_utm_dist(
                (cor_and_shif_cor[0][0], cor_and_shif_cor[0][1]),
                (cor_and_shif_cor[1][0], cor_and_shif_cor[1][1]),
            ),
            zip(coordinates, shifted_coordinates),
        )
        return list(distances)

    df["distances"] = df.apply(calc_dist, axis=1)
    df.drop("shifted_coordinates", axis=1, inplace=True)


def ms_to_kmh(speeds: list[float]) -> list[float]:
    """convert list of speeds in meters per second to list of speeds with
    km per second.

    Args:
        speeds (list[float]): list of speeds

    Returns:
        list[float]: new list of speeds
    """
    return [speed * 3.6 for speed in speeds]


def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate aggregate min, max, and avg for dataframe

    Args:
        df (pd.DataFrame): Dataframe to calculate on

    Returns:
        tuple[float, float, float]: tuple containing the avg, min and max
    """
    # per entry: max, min, avg speed across entire segment, median
    df["speeds"] = df["distances"].apply(ms_to_kmh)
    df["avg_speed"] = df["speeds"].apply(mean)  # type: ignore
    df["max_speed"] = df["speeds"].apply(max)
    df["min_speed"] = df["speeds"].apply(min)
    df["median"] = df["speeds"].apply(median)  # type: ignore

    # for all entries: median_avg, median_min, median_max
    median_avg = df["avg_speed"].median()
    median_min = df["min_speed"].median()
    median_max = df["max_speed"].median()

    return (median_avg, median_min, median_max)


def main():
    filename = "segment_10240935_linestring.json"
    df = create_df_from_json(filename)
    university_boulevard_osm_id = 10240935
    filtered_df = filter_segments(df, university_boulevard_osm_id)
    calculate_distance(filtered_df)
    avg, min, max = calculate_metrics(filtered_df)
    print(avg, min, max, sep=", ")

    # for (x, y, time), speed in zip(filtered_df.iloc[7]['coordinates'], filtered_df.iloc[7]['speeds']):
    #     print(time, speed, sep=", ")


if __name__ == "__main__":
    main()
