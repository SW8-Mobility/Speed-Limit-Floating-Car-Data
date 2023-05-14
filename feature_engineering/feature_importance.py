import joblib
from sklearn.inspection import permutation_importance
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
                f"{diabetes.feature_names[i]:<8}"
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

def main():
    model_folder = "/share-files/share-files/runs/05_12_12_18/"
    paths = glob.glob(model_folder + '*.joblib')

    formatter = SKFormatter(dataset="/share-files/share-files/raw_data_pkl/features_and_ground_truth_2012.pkl", dataset_size=1000, test_size=0.01)
    x_train, _, y_train, _ = formatter.generate_train_test_split()

    for path in paths[:1]:
        print(f"loading model: {path}")
        model = joblib.load(path)
        # model_members(model)
        calculate_feature_importance(model, x_train, y_train)

if __name__ == "__main__":
    main()
