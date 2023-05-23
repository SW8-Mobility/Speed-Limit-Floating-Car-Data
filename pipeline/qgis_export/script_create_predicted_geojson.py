import pandas as pd # type: ignore
from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.preprocessing.compute_features.feature import FeatureList, Feature
from pipeline.run_models import get_prediction, append_predictions_to_df
import joblib  # type: ignore


def main():
    dataset = pd.read_pickle("/share-files/raw_data_pkl/features_and_ground_truth_combined.pkl")

    xgboost = joblib.load("/share-files/models/xgboost_best_model.joblib")

    preds = get_prediction("xgboost", xgboost, dataset)

    dataset_with_preds = append_predictions_to_df(preds)

    pd.to_pickle("/share-files/raw_data_pkl/predictions.pkl")

    print(dataset_with_preds)



if __name__ == "__main__":
    main()
