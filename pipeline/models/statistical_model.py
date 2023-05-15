from pipeline.models.utils.scoring import quantize_results
import pandas as pd  # type: ignore
from pipeline.preprocessing.compute_features.feature import Feature
import numpy as np


class StatisticalModel:
    """Basic statistical model for predicting speed limits"""

    def predict(self, x: pd.DataFrame) -> pd.Series:
        """Simply return the aggregate median as the prediction.

        Args:
            x (pd.DataFrame): Datafame to predict on.

        Returns:
            pd.Series: list of predictions.
        """
        return quantize_results(x[Feature.AGGREGATE_MEDIAN.value].values)
