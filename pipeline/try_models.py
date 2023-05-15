from typing import Any
import pandas as pd 
import joblib 
import os
import glob 
from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.preprocessing.compute_features.feature import Feature

def load_models(model_folder: str) -> list[tuple[str, Any]]:
    """Load every model from a folder.

    Args:
        model_folder (str): path to the folder with models ie.
            "share-file/runs/folder1"

    Returns:
        list[tuple[str, Any]]: list with tuples of the name 
        to the model. 
    """

    model_paths = glob.glob(model_folder + "/*.joblib")
    # get filename of each path, only works on linux/unix
    names = [os.path.basename(name) for name in model_paths] 

    models = []
    for name, path in zip(names, model_paths):
        models.append((name, joblib.load(path)))

    return models

def per_road_type_df(pickle_path: str) -> dict[str, pd.DataFrame]:
    """loads dataset from path, splits the dataframe into dataframe for 
    each road type. 

    Args:
        filepath (str): path to pickled dataset 

    Returns:
        dict[str, pd.DataFrame]: road_type to dataframe
    """
    df = pd.read_pickle(pickle_path)
    road_types = list(df[Feature.VEJTYPESKILTET.value].unique())
    road_type_to_df: dict[str, pd.DataFrame] = {}
    for type in road_types:
        # select only rows for the specific road_type
        type_df = df[df[Feature.VEJTYPESKILTET.value] == type]  
        road_type_to_df[type] = type_df

    return road_type_to_df

def main():
    params = ...
    dataset_path = ""
    dfs = per_road_type_df(dataset)
    models_folder = ""
    load_models(models_folder)
    for road_type, df in dfs.items():
        skf = SKFormatter(df, **params, test_size=0.0001, full_dataset=True)
        x, _, y, _ = skf.generate_train_test_split()
        
    

if __name__ =='__main__':
    main()