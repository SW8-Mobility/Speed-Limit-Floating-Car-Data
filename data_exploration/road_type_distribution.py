import pandas as pd
from pipeline.preprocessing.compute_features.feature import Feature
df = pd.read_pickle('2012.pkl')

# remove duplicates 
cols = df.columns
df = df[cols].loc[df[cols].astype(str).drop_duplicates().index]

road_type_col = Feature.VEJTYPESKILTET.value
df = df.groupby([road_type_col])[road_type_col].count()

# convert to percentage
total_count = len(df)
road_dist_df = df.apply(lambda count: (count / total_count) * 100)

print(road_dist_df)