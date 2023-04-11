from typing import Any, Callable
import pandas as pd  # type: ignore

UnixTimestamp = int
Coordinate = tuple[float, float]
CoordinateWithTime = tuple[float, float, UnixTimestamp]
Trip = list[CoordinateWithTime]
ListOfSpeeds = list[float]
FeatureCalculator = Callable[[pd.DataFrame], Any]