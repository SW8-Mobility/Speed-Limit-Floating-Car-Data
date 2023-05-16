import pandas as pd
from pipeline.preprocessing.compute_features.feature import Feature

share_files = "/share-files/raw_data_pkl/"

paths = [
    share_files + "vejtype_2012.pkl",
    share_files + "vejtype_2013.pkl",
    share_files + "vejtype_2014.pkl",
]

for p in paths:
    df = pd.read_pickle(p)
    road_dist_df = df[Feature.VEJTYPESKILTET.value].value_counts(normalize=True) * 100
    road_amount_df = df[Feature.VEJTYPESKILTET.value].value_counts()

    print(road_dist_df)
    print(road_amount_df)
