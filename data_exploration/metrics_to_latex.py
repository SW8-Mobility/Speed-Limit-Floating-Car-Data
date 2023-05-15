import pandas as pd 

metrics_df = pd.read_csv("metric.csv")
metrics_df.to_latex("metrics_latex.txt")