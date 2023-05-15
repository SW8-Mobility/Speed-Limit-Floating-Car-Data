import pandas as pd 

metrics_df = pd.read_csv("metric.txt")
metrics_df['model'] = metrics_df['model'].apply(lambda model_name: model_name.replace('_', ' '))
metrics_cols = metrics_df.columns.drop(['model'])
for col in metrics_cols:
    metrics_df[col] = metrics_df[col].apply(lambda metric: str(round(float(metric), 2)))
metrics_df.to_latex("metrics_latex.txt", index=False)