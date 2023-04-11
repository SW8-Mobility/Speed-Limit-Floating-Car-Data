from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd # type: ignore

CORE_NUM = 16
RANDOM_STATE = 42

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from a CSV file and split it into features and target.

    Args:
        path (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of two dataframes: the features and target.
    """
    df = pd.read_csv(path)
    y = df['speed_limit']
    X = df.drop(columns=['speed_limit'])
    return X, y


def create_mlp(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[MLPClassifier, dict]:
    """
    Create and tune the hyperparameters of an MLP classifier using GridSearchCV with k-fold cross-validation.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        tuple[MLPClassifier, dict]: A tuple of the best MLP classifier and the best hyperparameters.
    """
    # create pipeline with mlp and scaler
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=42))
    ])

    # define the hyperparameters to search over
    parameters = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__activation': ['relu', 'logistic'],
        'mlp__solver': ['adam', 'lbfgs'],
        'mlp__learning_rate': ['constant', 'adaptive'],
        'mlp__early_stopping': [True, False],
        'mlp__tol': [1e-3, 1e-4, 1e-5],
        'mlp__beta_1': [0.9, 0.8, 0.7],
        'mlp__beta_2': [0.999, 0.9, 0.8],
        'mlp__validation_fraction': [0.1, 0.2, 0.3]
    }

   # perform grid search with k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(pipeline, parameters, cv=kfold, n_jobs=CORE_NUM)
    grid_search.fit(x_train, y_train)

    # return the best random forest model and the best hyperparameters found by the grid search
    return grid_search.best_estimator_, grid_search.best_params_ # type: ignore


def evaluate_model(model: MLPClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate a MLPClassifier classifier on the testing data using F1-score.

    Args:
        model (MLPClassifier): The MLP classifier to evaluate.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        float: The F1-score of the model on the testing data.
    """
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='micro') # type: ignore


if __name__ == '__main__':
    # load the data
    x, y = load_data('path')
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
