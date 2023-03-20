import sys
import pandas as pd

path_root = "data/pickle_files"
from wrap_up import map_segments_to_coordinates

# :)
segment_to_coordinates: dict[str, list[int, int, int]] = dict()
errors = []


def clean_df(df: pd.DataFrame) -> None:
    """clean
    basically, remove the null values from osm_id and its corresponding coordinates
    also wrong lmao

    Args:
        df (pd.DataFrame): _description_
    """
    combined_col = df.apply(lambda d: list(zip(d["osm_id"], d["coordinates"])), axis=1)
    print(combined_col.iloc[0])
    combined_col.apply(lambda sc: list(filter(lambda elem: elem[0] is not None, sc)))
    # print(combined_col.iloc[0])
    segments, coordinates = list(zip(*combined_col))
    df["coordinates"] = coordinates
    df["osm_id"] = segments


def map_segments(segments, coordinates):
    try:
        return map_segments_to_coordinates(segments, coordinates)
    except Exception as e:
        print(e)
        return None

def append_coordinates(key_coordinates: list[tuple[int, list]]) -> None:
    if key_coordinates is None:
        print(key_coordinates)
        return None  # Dont know why some are None
    for mapped_cords in key_coordinates:
        key, coordinates = mapped_cords
        if key not in segment_to_coordinates:
            segment_to_coordinates[key] = []
        else:
            segment_to_coordinates[key].extend(coordinates)
        return key_coordinates

def create_segment_to_coordinate_df(df: pd.DataFrame) -> pd.DataFrame:
    mapped_coordinates: pd.Series = df.apply(
        lambda d: map_segments_to_coordinates(d["osm_id"], d["coordinates"]), axis=1
    )

    mapped_coordinates.apply(lambda seg_and_cor: append_coordinates(seg_and_cor))
    l = [(k, v) for k, v in segment_to_coordinates.items()] # convert dictionary to list 
    mapped_df = pd.DataFrame(l, columns=["segment", "coordinates"])
    return mapped_df

def main():
    sys.setrecursionlimit(10000)
    df: pd.DataFrame = (
        pd.read_pickle("C:\\Users\\ax111\\Documents\\Personal documents\\Coding\\SW8\\geo_json_metrics\\geo_json_metrics\\data\\pickle_files\\2012.pkl").infer_objects().head(1000)
    )
    mapped_df = create_segment_to_coordinate_df(df)
    sys.setrecursionlimit(1000) # reset 
    mapped_df.to_csv('test.csv', index=False)

if __name__ == "__main__":
    main()
