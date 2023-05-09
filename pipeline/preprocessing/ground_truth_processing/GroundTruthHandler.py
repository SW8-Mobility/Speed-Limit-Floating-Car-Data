from io import TextIOWrapper
import json
import pandas as pd # type: ignore
import os

class GroundTruthHandler:

    def __init__(self):
        pass

    @classmethod
    def load_from_geojson(cls, file: TextIOWrapper, year: int):
        data = json.load(file)
        return cls.__clean_vejman_data(data, year)

    @classmethod
    def __clean_vejman_data(cls, df: pd.DataFrame, year: int) -> pd.DataFrame:

        # Extract the properties and remove the linestring information
        cleandf = pd.DataFrame([x['properties'] for x in df['features']])

        # Only use speed limit, that have not been changed after the specified year
        cleandf = cleandf.loc[
            cleandf["HAST_SENEST_RETTET"] < f"{year+1}-01-01T00:00:00"
        ]

        for index, val in cleandf.iterrows():
            print(val['HAST_SENEST_RETTET'])

        # Only keep the data we need along with removing all data without speed limits known
        cleandf = cleandf[['osm_id', "KODE_HAST_GENEREL_HAST"]]
        cleandf = cleandf.dropna()
        
        # Type osm_id and speed limit as int
        cleandf = cleandf.astype(int)

        return cleandf

    @classmethod 
    def add_ground_truth_to_FCD_Data(cls, fcd_data: pd.DataFrame, ground_truth_data: pd.DataFrame):
        return pd.merge(fcd_data, ground_truth_data, on="osm_id")


def main():
    with open(os.path.dirname(__file__) + "/osm_vejman_merged_intersects_no_dupes_no_nulls.geojson") as file:
        data = GroundTruthHandler.load_from_geojson(file, 2014)
        print(data)

        
if __name__ == "__main__":
    main()
