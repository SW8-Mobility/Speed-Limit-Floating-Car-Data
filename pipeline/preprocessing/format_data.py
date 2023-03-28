import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)

import pandas as pd
from pipeline.preprocessing.feature import Feature

path_root = "data/pickle_files"
errors: list[str] = []
coordinates_with_time = tuple[int, int, int]
trip = list[coordinates_with_time]

# types for refactoring
segmemt_to_coordinate_dict = dict[int, list[trip]]
trip_with_features = dict[Feature, object]  # feature and the values for the feature
segment_to_trip_dict = dict[int, list[trip_with_features]]


def clean_df(df: pd.DataFrame) -> None:
    """clean
    basically, remove the null values from osm_id and its corresponding coordinates
    also wrong x

    Args:
        df (pd.DataFrame): _description_
    """
    combined_col = df.apply(lambda d: list(zip(d["osm_id"], d["coordinates"])), axis=1)
    combined_col.apply(lambda sc: list(filter(lambda elem: elem[0] is not None, sc)))
    segments, coordinates = list(zip(*combined_col))
    df["coordinates"] = coordinates
    df["osm_id"] = segments


def append_coordinates(
    key_coordinates: list[tuple[int, list]], segment_dict: segmemt_to_coordinate_dict
) -> None:
    if key_coordinates is None:
        return  # Dont know why some are None

    for mapped_cords in key_coordinates:
        key, coordinates = mapped_cords
        if key not in segment_dict:
            segment_dict[key] = [coordinates]
        else:
            segment_dict[key].extend([coordinates])


def map_segments(segments, coordinates):
    try:
        return map_segments_to_coordinates(segments, coordinates)
    except Exception as e:
        print(e)
        return None


def create_segment_to_coordinate_df(df: pd.DataFrame) -> pd.DataFrame:
    segment_to_coordinates: segmemt_to_coordinate_dict = dict()

    """ 
    step 1: for each trip, seggregate the coordinates, according to the segment (osm_id)
    so for each trip, something like [(segment1, coordinates), (segment2, coordinates)]
    this will happen for each trip, so thefore a series.
    """
    mapped_coordinates: pd.Series = df.apply(
        lambda d: map_segments_to_coordinates(d["osm_id"], d["coordinates"]), axis=1
    )

    """
    step 2: 
    """
    mapped_coordinates.apply(
        lambda seg_and_cor: append_coordinates(seg_and_cor, segment_to_coordinates)  # type: ignore
    )

    # create dataframe from the dictionary
    l = [
        (k, v) for k, v in segment_to_coordinates.items()
    ]  # convert dictionary to list
    mapped_df = pd.DataFrame(l, columns=["osm_id", "coordinates"])
    return mapped_df


segment_to_coordinates_list = list[tuple[int, list]]


def map_segments_to_coordinates(
    segments: list, coordinates: list
) -> segment_to_coordinates_list:
    """Aggregate lists of segments and coordinates, such that coordinates are
    associated with its corresponding segment id.

    Args:
        segments (list): _description_
        coordinates (list): _description_

    Returns:
        segment_to_coordinates_list: _description_
    """
    if len(segments) == 0:
        return []

    result = []
    current_seg = (segments[0], [])
    for seg, cor in zip(segments, coordinates):
        if seg != current_seg[0]:  # new segment starts
            result.append(current_seg)
            current_seg = (seg, [cor])
        else:
            current_seg[1].append(cor)

    result.append(current_seg)

    return result


def main():
    df: pd.DataFrame = (
        pd.read_pickle(
            "C:\\Users\\ax111\\Documents\\Personal documents\\Coding\\SW8\\geo_json_metrics\\geo_json_metrics\\data\\pickle_files\\2012.pkl"
        )
        .infer_objects()
        .head(1000)
    )
    mapped_df = create_segment_to_coordinate_df(df)
    mapped_df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
