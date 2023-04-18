import pandas as pd
from pipeline.preprocessing.ground_truth_processing.process_merged_geojson import create_df_from_merged_osm_vejman
def format_hast_generel_hast(df: pd.DataFrame) -> None:
    mapping = {
        '130 - Motorvej': 130,
        '50 - Indenfor byzonetavler': 50,
        '80 - Udenfor byzonetavler': 80
    }
    df['hast_generel_hast'] = df['hast_generel_hast'].apply(lambda hast: mapping[hast])

    print(df['hast_generel_hast'].head())


def prepare_ground_truth(df: pd.DataFrame) -> None:
    df.drop(['kode_vejstiklasse', 'kode_vejtypeskiltet', 'kode_hast_generel_hast'], inplace=True, axis=1)
    df = df[df['hast_generel_hast'].notna()]
    df = df[df['hast_gaeldende_hast'].notna()]

    format_hast_generel_hast(df)

    df['cpr_vejnavn'] = df['vejtypeskiltet'].astype(str)
    df['vejtypeskiltet'] = df['vejtypeskiltet'].astype(str)
    df['vejtypeskiltet'] = df['vejtypeskiltet'].astype(str)
    df['vejtypeskiltet'] = df['vejtypeskiltet'].astype(str)
    df['hast_gaeldende_hast'] = df['hast_gaeldende_hast'].astype(int)
    print(df.dtypes)
    print(df['hast_generel_hast'].head())

def join_ground_truth(df: pd.DataFrame, dataframe_year: int, ground_truth_path: str) -> None:
    # headers: ['osm_id', 'cpr_vejnavn', 'hast_generel_hast', 'kode_hast_generel_hast',
    #        'hast_gaeldende_hast', 'vejstiklasse', 'kode_vejstiklasse',
    #        'vejtypeskiltet', 'kode_vejtypeskiltet', 'hast_senest_rettet',
    #        'coordinates']
    ground_truth_df = pd.read_pickle(ground_truth_path)
    prepare_ground_truth(ground_truth_df)


    return ground_truth_df


    # pd.merge(df, ground_truth_df, on="osm_id")

if __name__ == '__main__':
    df = join_ground_truth(None, 2012, "/home/kubbe/speed_limit_floating_car_data/pipeline/data/pkl_files/ground_truth.pkl")