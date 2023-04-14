from pipeline.preprocessing.format_data import create_segment_to_coordinate_df
from pipeline.preprocessing.process_merged_geojson import create_df_from_merged_osm_vejman
import pandas as pd

def main():
    # Use ground truth
    filename = "/qgis_data/join_by_nearest_max_distance_2.geojson"
    df = create_df_from_merged_osm_vejman(filename)
    df.to_pickle("join_by_nearest_max_distance_2.pkl")

    # Format raw data
    df = pd.read_pickle(...).infer_objects()
    mapped_df = create_segment_to_coordinate_df(df)
    mapped_df.to_pickle("raw_data.pkl")

if __name__ == "__main__":
    main()
