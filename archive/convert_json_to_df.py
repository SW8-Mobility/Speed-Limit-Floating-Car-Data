""" Script used to convert all our json filed we fetched from the FCD api
and convert them into 3 dataframe. One for each year. 
"""

import pandas as pd  # type: ignore
import glob
from pipeline.preprocessing.df_processing import create_df_from_json
# from archive.gps_metrics import create_df_from_json
import tqdm

# gather all json files
paths_2012 = glob.glob("data_2012/*.json")
paths_2013 = glob.glob("data_2013/*.json")
paths_2014 = glob.glob("data_2014/*.json")

# create a dataframe for each file
dfs = []
for path in tqdm.tqdm(paths_2012):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0) # combine the dataframe
concat.to_pickle("pickle_files/2012.pkl") # create the pickle file

# same for 2013
dfs = []
for path in tqdm.tqdm(paths_2013):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0)
concat.to_pickle("pickle_files/2013.pkl")

# same for 2014
dfs = []
for path in tqdm.tqdm(paths_2014):
    dfs.append(create_df_from_json(path))
concat = pd.concat(dfs, axis=0)
concat.to_pickle("pickle_files/2014.pkl")
