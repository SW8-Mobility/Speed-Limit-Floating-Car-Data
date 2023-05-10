from pipeline.models.best_params_models import create_xgboost_best_params

def test_models_with_different_data():
    # https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
    from sklearn import datasets # type: ignore
    from sklearn.model_selection import train_test_split # type: ignore
    from sklearn.ensemble import RandomForestClassifier # type: ignore
    import pandas as pd
    from sklearn import metrics # type: ignore
    from sklearn.metrics import mean_squared_error # type: ignore

    # Loading the iris plants dataset (classification)
    iris = datasets.load_iris()
    print(iris.target_names)
    print(iris.feature_names)

    # dividing the datasets into two parts i.e. training datasets and test datasets
    X, y = datasets.load_iris(return_X_y=True)

    # Splitting arrays or matrices into random train and test subsets
    # i.e. 70 % training dataset and 30 % test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Model to test:
    model = create_xgboost_best_params(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics are used to find accuracy or error
    print()
    print("ACCURACY OF THE MODEL: ", mean_squared_error(y_test, y_pred, squared=False))