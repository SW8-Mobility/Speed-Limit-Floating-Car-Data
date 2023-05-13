import pandas as pd
from pipeline.preprocessing.compute_features.feature import Feature

share_files = "/share-files/pickle_files_features_and_ground_truth/"

paths = [
    share_files + "2012.pkl",
    share_files + "2013.pkl",
    share_files + "2014.pkl",
]

for p in paths:
    df = pd.read_pickle(p)
    # remove duplicates
    cols = df.columns
    df = df[cols].loc[df[cols].astype(str).drop_duplicates().index]

    road_type_col = Feature.VEJTYPESKILTET.value
    df = df.groupby([road_type_col])[road_type_col].count()

    # convert to percentage
    total_count = df.sum()
    road_dist_df = df.apply(lambda count: (count / total_count) * 100)

    print(road_dist_df)
