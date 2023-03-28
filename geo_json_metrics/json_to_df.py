import pandas as pd  # type: ignore
import glob
from geo_json_metrics.gps_metrics import create_df_from_json
import tqdm

paths_2012 = glob.glob("data_2012/*.json")
paths_2013 = glob.glob("data_2013/*.json")
paths_2014 = glob.glob("data_2014/*.json")

dfs = []
for path in tqdm.tqdm(paths_2012):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0)
concat.to_pickle("pickle_files/2012.pkl")

dfs = []
for path in tqdm.tqdm(paths_2013):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0)
concat.to_pickle("pickle_files/2013.pkl")

dfs = []
for path in tqdm.tqdm(paths_2014):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0)
concat.to_pickle("pickle_files/2014.pkl")
