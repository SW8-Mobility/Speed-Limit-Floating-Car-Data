from pandas import DataFrame, Series # type: ignore
from typing import Any
from io import open
import json
from pipeline.preprocessing.compute_features.type_alias import Trip

SegmentToCoordinateDict = dict[int, list[Trip]]
SegmentToCoordinatesList = list[tuple[int, list]]


class FCD_Formatter:
    @classmethod
    def from_json_file(self, file_path: str) -> DataFrame:
        """Loads a file and calls from_json_string on it

        Args:
            fcd_formatter (FCD_Formatter): The class instance
            file_path (str): Path to a json file

        Returns:
            DataFrame: _description_
        """
        with open(file_path, "r") as data_from_fcd:
            return self.from_json_string(data_from_fcd.read())

    @staticmethod
    def from_json_string(json_string: str) -> DataFrame:
        data = json.loads(json_string)  # Load from string
        data = _remove_fcd_request_wrapper(
            data
        )  # Remove the outer layer of json object
        data = _create_segment_to_coordinate_df(
            data
        )  # Convert from geojson format to osm_id -> trips

        return data


def _remove_fcd_request_wrapper(wrapdata: Any) -> DataFrame:
    """This function removes the FCD Wrapper that AAU FCD data comes in when it is requested.

    Args:
        wrapdata (Any): The json object from the AAU FCD Request

    Returns:
        DataFrame: the "meat" of the FCD Data
    """
    df = DataFrame.from_records(  # convert json file to dataframe
        wrapdata["results"]["features"]
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

    return df.infer_objects()  # infer types in dataframes


def _create_segment_to_coordinate_df(df: DataFrame) -> DataFrame:
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
    _clean_df(df)

    # step 1:
    # for each trip, seggregate the coordinates, according to the segment (osm_id)
    # so for each trip, something like [(segment1, coordinates), (segment2, coordinates)]
    # this will happen for each trip, so thefore a series.

    mapped_coordinates: Series = df.apply(
        lambda d: _map_segments_to_coordinates(d["osm_id"], d["coordinates"]), axis=1
    )

    # step 2:
    # For each trip, add to coordinates to the correct segment in the segment_to_coordinates dict

    mapped_coordinates.apply(
        lambda seg_and_cor: _append_coordinates(seg_and_cor, segment_to_coordinates)
    )

    # step 3:
    # create the output dataframe from the dictionary
    l = [
        (k, v) for k, v in segment_to_coordinates.items()  # converts dictionary to list
    ]

    mapped_df = DataFrame(l, columns=["osm_id", "coordinates"])
    return mapped_df


def _map_segments_to_coordinates(
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


def _append_coordinates(
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


def _clean_df(df: DataFrame) -> None:
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


def main():
    check = FCD_Formatter.from_json_file("data/2014_1.json")

    print(check)


if __name__ == "__main__":
    main()
