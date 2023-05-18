import joblib
from sklearn.inspection import permutation_importance

from pipeline.preprocessing.compute_features.feature import FeatureList, Feature
from pipeline.preprocessing.sk_formatter import SKFormatter
import glob

# ------------------------EXAMPLE-----------------------------
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
def example_func_feature_importance():
    diabetes = load_diabetes()
    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )

    model = Ridge(alpha=1e-2).fit(X_train, y_train)

    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(
                f"{model.fe:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}"
            )
# ------------------------------------------------------------

def model_members(model):
    print(model.n_features_in_)
    print(model.estimators_)
    print(model.feature_importances_)
    print(model.feature_names_in_)
# ------------------------------------------------------------


def calculate_feature_importance(model, X_val, y_val) -> None:
    """Prints to the console the feature importance, grouped by the scoring metrics.

    Args:
        r_multi: A dict-like object that contains; mean importance, importance standard deviation and feature importance for every feature.
    """
    scoring = ["r2", "neg_mean_absolute_percentage_error", "neg_mean_squared_error"]
    print(model)
    r_multi = permutation_importance(
        model, X_val, y_val, random_state=0, scoring=scoring, n_jobs=15
    )

    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(
                    # f"{formatter.params.get('processed_df_columns'):<8}"
                    f"{r.importances_mean[i]:.3f}"
                    f" +/- {r.importances_std[i]:.3f}"
                )

def calc_permutation_importance(model, x, y) -> None:
    result = permutation_importance(model, x, y, n_repeats=10, random_state=0)
    shifted_res = [i + abs(min(result.importances_mean)) for i in result.importances_mean]
    min_importance = min(shifted_res)
    max_importance = max(shifted_res)
    normalized_importances = [(imp - min_importance) / (max_importance - min_importance) for imp in shifted_res]
    n = 20 # Divide by 7 due to having 7 array features
    column_names = [Feature.AGGREGATE_MIN.value, Feature.AGGREGATE_MAX.value, Feature.AGGREGATE_MEAN.value, Feature.AGGREGATE_MEDIAN.value, Feature.MEANS.value, Feature.MINS.value, Feature.MAXS.value, Feature.MEDIANS.value, Feature.SPEEDS.value, Feature.ROLLING_AVERAGES.value, Feature.VCR.value]
    averaged_values = []
    i = 0
    while i <= len(normalized_importances):
        if i >= 4:
            average = sum(normalized_importances[i:i + n]) / n
        else:
            average = normalized_importances[i]
        averaged_values.append(average)
        i += n if i >= 4 else 1

    result = list(zip(column_names, averaged_values))
    for res in result:
        print(res)


def main():
    model_folder = "/share-files/runs/05_15_14_27/"

    formatter = SKFormatter(dataset="/share-files/raw_data_pkl/features_and_ground_truth_combined.pkl", full_dataset=True, test_size=0.25, discard_features=FeatureList(
            [
                Feature.OSM_ID,
                Feature.COORDINATES,
                Feature.DISTANCES,
            ]
        ))
    x_train, _, y_train, _ = formatter.generate_train_test_split()

#    model = joblib.load(model_folder + '05_15_14_27_mlp.joblib')
#    calc_permutation_importance(model, x_train, y_train)

#    model = joblib.load(model_folder + '05_15_14_27_random_forest.joblib')
#    calc_permutation_importance(model, x_train, y_train)

    model = joblib.load(model_folder + '05_15_14_27_xgboost.joblib')
    r = permutation_importance(model, x_train, y_train, n_repeats=30, random_state=0)

    cols = list(x_train.columns)
    for i in r.importances_mean.argsort()[::-1]:
        #if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(
            f"{cols[i][1]:<8}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}"
        )


#    model = joblib.load(model_folder + '05_15_14_27_logistic_regression.joblib')
#    calc_permutation_importance(model, x_train, y_train)



if __name__ == "__main__":
    main()
