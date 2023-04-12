from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd

from pipeline.models.utils.find_closest_speed_limit import load_data  # type: ignore

CORE_NUM = 16
RANDOM_STATE = 42

def create_mlp_grid_search(
    x_train: pd.DataFrame, y_train: pd.Series, k: int = 5
) -> tuple[MLPClassifier, dict]:
    """
    Create and tune the hyperparameters of an MLP classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best MLP classifier and the best hyperparameters.
    """
    # create pipeline with mlp and scaler
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=1000, random_state=42)),
        ]
    )

    # define the hyperparameters to search over
    parameters = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__activation": ["relu", "logistic"],
        "mlp__solver": ["adam", "lbfgs"],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__early_stopping": [True, False],
        "mlp__tol": [1e-3, 1e-4, 1e-5],
        "mlp__beta_1": [0.9, 0.8, 0.7],
        "mlp__beta_2": [0.999, 0.9, 0.8],
        "mlp__validation_fraction": [0.1, 0.2, 0.3],
    }

    # perform grid search with k-fold cross-validation
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(pipeline, parameters, cv=kfold, n_jobs=CORE_NUM)
    grid_search.fit(x_train, y_train)

    # return the best random forest model and the best hyperparameters found by the grid search
    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


def random_forest_regressor_gridsearch(
    X_train: pd.DataFrame, y_train: pd.DataFrame, k: int = 5
) -> tuple[RandomForestRegressor, dict]:
    # Create a random forest regressor
    rfr = RandomForestRegressor(random_state=42)

    # Set up the parameter grid to search over
    param_grid = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [5, 10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt", "log2"],
    }

    # Create the grid search object
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rfr, param_grid=param_grid, cv=kfold, n_jobs=CORE_NUM
    )

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_  # type: ignore


if __name__ == "__main__":
    pass
