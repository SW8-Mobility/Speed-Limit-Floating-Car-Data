from typing import Any, Callable
import pandas as pd
UnixTimestamp = int
CoordinateWithTime = tuple[float, float, UnixTimestamp]
Trip = list[CoordinateWithTime]
ListOfSpeeds = list[list[float]]
row = pd.DataFrame
FeatureCalculator = Callable[[row], Any]