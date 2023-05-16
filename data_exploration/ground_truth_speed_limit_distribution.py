import pandas as pd
from pipeline.preprocessing.compute_features.feature import Feature

share_files = "/share-files/raw_data_pkl/"

path = "features_and_ground_truth_combined.pkl"

df = pd.read_pickle(share_files + path)
speed_dist_df = df[Feature.HAST_GAELDENDE_HAST.value].value_counts(normalize=True) * 100
speed_amount_df = df[Feature.HAST_GAELDENDE_HAST.value].value_counts()
print("---- SPEED DISTRIBUTION ----")
print(speed_dist_df)
print("---- SPEED AMOUNT ----")
print(speed_amount_df)
