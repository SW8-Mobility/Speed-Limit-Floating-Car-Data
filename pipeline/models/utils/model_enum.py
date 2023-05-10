from __future__ import annotations
from enum import Enum


class Model(Enum):
    MLP = "mlp"
    RF = "random_forest"
    XGB = "xgboost"
    LOGREG = "logistic_regression"
    STATMODEL = "statistical_model"

    @staticmethod
    def regression_models_names() -> list[str]:
        return [Model.LOGREG.value, Model.RF.value]

    def __str__(self) -> str:
        return self.value
