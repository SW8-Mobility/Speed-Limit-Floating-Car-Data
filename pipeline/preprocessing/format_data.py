import sys
import os
from typing import Any


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grand_parent = os.path.dirname(parent)
sys.path.append(grand_parent)

import pandas as pd  # type: ignore
from pipeline.preprocessing.compute_features.type_alias import Trip

path_root = "data/pickle_files"

SegmentToCoordinateDict = dict[int, list[Trip]]
SegmentToCoordinatesList = list[tuple[int, list]]


def clean_df(df: pd.DataFrame) -> None:
    """Remove None values from trips. Some trips have None values
    in the osm_id list. Remove these, and the corresponding coordinate
    values.

    Args:
        df (pd.DataFrame): dataframe with trips
    """
    combined_col = df.apply(  # combine osm_id's and coordinates for each trip
        lambda d: list(zip(d["osm_id"], d["coordinates"])), axis=1
    )
    combined_col = combined_col.apply(  # filter the None values
        lambda sc: list(filter(lambda elem: elem[0] is not None, sc))
    )
    df["osm_id"] = combined_col.apply(  # get osm_id from tuple list
        lambda d: [elem[0] for elem in d]
    )
    df["coordinates"] = combined_col.apply(  # get coordinates from tuple list
        lambda d: [elem[1] for elem in d]
    )


def append_coordinates(
    osm_and_coordinates: list[tuple[int, list]], segment_dict: SegmentToCoordinateDict
) -> None:
    """Appends coordinates from a trip to the correct segment in segment_dict

    Args:
        key_coordinates (list[tuple[int, list]]): list of segments and coordinates from each trip
        segment_dict (SegmemtToCoordinateDict): dict to append to
    """
    for osm, coordinates in osm_and_coordinates:
        if osm not in segment_dict:
            segment_dict[osm] = [coordinates]
        else:
            segment_dict[osm].extend([coordinates])


def create_segment_to_coordinate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Main method for converting our dataframe of a trip per row to a dataframe
    of a segment per row.

    Args:
        df (pd.DataFrame): dataframe with trips

    Returns:
        pd.DataFrame: dataframe with each segment as a row
    """

    segment_to_coordinates: SegmentToCoordinateDict = dict()

    # step 0:
    # remove None values
    clean_df(df)

    # step 1:
    # for each trip, seggregate the coordinates, according to the segment (osm_id)
    # so for each trip, something like [(segment1, coordinates), (segment2, coordinates)]
    # this will happen for each trip, so thefore a series.

    mapped_coordinates: pd.Series = df.apply(
        lambda d: map_segments_to_coordinates(d["osm_id"], d["coordinates"]), axis=1
    )

    # step 2:
    # For each trip, add to coordinates to the correct segment in the segment_to_coordinates dict

    mapped_coordinates.apply(
        lambda seg_and_cor: append_coordinates(seg_and_cor, segment_to_coordinates)  # type: ignore
    )

    # step 3:
    # create the output dataframe from the dictionary
    l = [
        (k, v) for k, v in segment_to_coordinates.items()  # converts dictionary to list
    ]
    mapped_df = pd.DataFrame(l, columns=["osm_id", "coordinates"])
    return mapped_df


def map_segments_to_coordinates(
    segments: list, coordinates: list
) -> SegmentToCoordinatesList:
    """Aggregate lists of segments (osm_id) and coordinates for a trip, such that coordinates are
    associated with its corresponding segment id. The reason why this works is that there is an
    equal amount of osm id's and coordinates, e.g [osm_1, osm_1, osm_2] and [coor1, coor2, coor3]

    Args:
        segments (list): list of osm_ids
        coordinates (list): list of coordinates

    Returns:
        segment_to_coordinates_list: list of tuples with a segment (osm_id), and it's coordinates as a list
    """
    if len(segments) == 0:
        return []

    result = []
    current_seg: tuple[int, Any] = (segments[0], [])
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
    mapped_df.to_pickle("test.pkl")


if __name__ == "__main__":
    main()
