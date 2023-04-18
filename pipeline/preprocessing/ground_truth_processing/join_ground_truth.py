import pandas as pd

# from pipeline.preprocessing.ground_truth_processing.process_merged_geojson import create_df_from_merged_osm_vejman


def format_hast_generel_hast(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "130 - Motorvej": 130,
        "50 - Indenfor byzonetavler": 50,
        "80 - Udenfor byzonetavler": 80,
    }
    df["hast_generel_hast"] = df["hast_generel_hast"].apply(lambda hast: mapping[hast])
    return df


def prepare_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        ["kode_vejstiklasse", "kode_vejtypeskiltet", "kode_hast_generel_hast"], axis=1
    )
    df = df[df["hast_generel_hast"].notna()]
    df = df[df["hast_gaeldende_hast"].notna()]
    df = df.fillna(value="None")  # make None values strings

    df = format_hast_generel_hast(df)

    df = df.astype(
        {"cpr_vejnavn": str, "vejtypeskiltet": str, "hast_gaeldende_hast": int}
    )
    df["hast_senest_rettet"] = pd.to_datetime(
        df["hast_senest_rettet"], format="%Y/%m/%d %H:%M:%S"
    )


    return df


def join_ground_truth(
    df: pd.DataFrame, dataframe_year: int, ground_truth_path: str
) -> pd.DataFrame:
    # headers: ['osm_id', 'cpr_vejnavn', 'hast_generel_hast', 'kode_hast_generel_hast',
    #           'hast_gaeldende_hast', 'vejstiklasse', 'kode_vejstiklasse',
    #           'vejtypeskiltet', 'kode_vejtypeskiltet', 'hast_senest_rettet',
    #           'coordinates']
    ground_truth_df = pd.read_pickle(ground_truth_path)
    vejman_features = prepare_ground_truth(ground_truth_df)

    # only use speed limit, that have not been changed after the year for the fcd data
    vejman_features = vejman_features.loc[vejman_features["hast_senest_rettet"] < f"{dataframe_year+1}/01/01 00:00:00"]

    df = pd.merge(df, vejman_features, on="osm_id")

    return df



if __name__ == "__main__":
    df = join_ground_truth(
        None,
        2012,
        "C:/Users/ax111/Documents/Personal documents/SW8/speed_limit_floating_car_data/pipeline/preprocessing/ground_truth_processing/ground_truth.pkl",
    )
    print(df["hast_generel_hast"].head())
    print(df.dtypes)
