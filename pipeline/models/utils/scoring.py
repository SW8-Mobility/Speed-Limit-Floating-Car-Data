from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
import numpy as np

def mean_absolute_percentage_error(ground_truth, prediction):
        return np.mean(np.abs((ground_truth - prediction) / ground_truth)) * 100

def score_model(ground_truth, prediction) -> tuple[float, float, float, float, float, float]:
    # Assuming y_true and y_pred are your true and predicted target variables, respectively
    mae = mean_absolute_error(ground_truth, prediction)
    mape = mean_absolute_percentage_error(ground_truth, prediction)
    mse = mean_squared_error(ground_truth, prediction)
    rmse = mean_squared_error(ground_truth, prediction, squared=False)
    r2 = r2_score(ground_truth, prediction)
    ev = explained_variance_score(ground_truth, prediction)
    return mae, mape, mse, rmse, r2, ev # type: ignore