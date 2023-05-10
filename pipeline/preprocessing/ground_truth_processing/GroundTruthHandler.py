from io import TextIOWrapper
import json
import pandas as pd  # type: ignore
import os


class GroundTruthHandler:
    """This class handles formatting the "dirty" ground truth dataset and
    also handles merging the ground truth with the FCD dataset for use in model training
    """

    @classmethod
    def load_from_geojson(cls, file: TextIOWrapper, year: int) -> pd.DataFrame:
        """Loads the ground truth from the dirty geojson and cleans it

        Args:
            file (TextIOWrapper): the file of the ground truth geojson.
            year (int): The year that the ground truth speed limit is from at the latest.

        Returns:
            pd.DataFrame: A dataframe that has 2 coloumns: osm_id and HAST_GAELDENDE_HAST, the latter being the speed limit
        """
        data = json.load(file)
        return cls.__clean_vejman_data(data, year)

    @classmethod
    def __clean_vejman_data(cls, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Cleans the dataframe by removing the unnecessary columns along
            with filtering speed limits that have changed after the given year.

        Args:
            df (pd.DataFrame): unclean ground truth data in dataframe format
            year (int): The year that the ground truth speed limit is from at the latest.

        Returns:
            pd.DataFrame: A dataframe that has 2 coloumns: osm_id and HAST_GAELDENDE_HAST, the latter being the speed limit
        """
        # Extract the properties and remove the linestring information
        cleandf = pd.DataFrame([x["properties"] for x in df["features"]])

        # Only use speed limit, that have not been changed after the specified year
        cleandf = cleandf.loc[
            cleandf["HAST_SENEST_RETTET"] < f"{year+1}-01-01T00:00:00"
        ]

        # Only keep the data we need along with removing all data without speed limits known
        cleandf = cleandf[["osm_id", "HAST_GAELDENDE_HAST"]]
        cleandf = cleandf.dropna()

        # Type both osm_id and speed limit as int
        cleandf = cleandf.astype(int)

        return cleandf

    @classmethod
    def add_ground_truth_to_FCD_Data(
        cls, fcd_data: pd.DataFrame, ground_truth_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Adds the ground truth speed limit as a coloumn HAST_GAELDENDE_HAST to a dataframe by matching the osm_id coloumn.

        Args:
            fcd_data (pd.DataFrame): FCD data with osm_id coloumn
            ground_truth_data (pd.DataFrame): Clean ground truth speed limit data with osm_id and HAST_GAELDENDE_HAST coloumns

        Returns:
            pd.DataFrame: Dataframe which has the same coloumns as fcd_data input but with HAST_GAELDENDE_HAST coloumn added.
        """
        return pd.merge(fcd_data, ground_truth_data, on="osm_id")


def main():
    with open(
        os.path.dirname(__file__)
        + "/osm_vejman_merged_intersects_no_dupes_no_nulls.geojson"
    ) as file:
        data = GroundTruthHandler.load_from_geojson(file, 2014)
        print(data)


if __name__ == "__main__":
    main()
