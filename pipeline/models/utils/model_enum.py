from enum import Enum


class ModelEnum(Enum):
    MLP = "mlp"
    RF = "random forest"
    XGB = "xgboost"
    LOGREG = "logistic regression"
    STATMODEL = "statistical model"

    def __str__(self) -> str:
        return self.value
