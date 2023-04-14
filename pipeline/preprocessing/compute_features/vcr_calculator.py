import pandas as pd  # type: ignore
from pipeline.preprocessing.compute_features.feature import Feature


class TooFewElementsException(Exception):
    """Raised when there are too few elements in the input list"""

    pass


def multiple_trips_vcr(row: pd.DataFrame) -> list[list[float]]:
    """Compute vcr for each trip in the speeds col.

    Args:
        row (pd.DataFrame): row with speeds col

    Returns:
        list[list[float]]: vcr for each trip [[speeds],[speeds]] -> [[vcr's],[vcr's]]
    """
    return [map_vcr(trip) for trip in row[Feature.SPEEDS.value] if len(trip) >= 2]


def map_vcr(velocities: list[float]) -> list[float]:
    """Map the VCR function over each pair from the list, skipping the first element
    Args:
        velocities (list[float]): A list of floats corresponding to the velocity (technically the speed since there is no
        vector describing the direction)
    Returns:
        list[float]: new list of Velocity Change Rates (the list is length of inputlist - 1)
    """

    if len(velocities) < 2:
        raise TooFewElementsException

    # list[:-1] all elements except the last one
    # list[1:] all elements except the first one
    return [vcr(v1, v2) for v1, v2 in zip(velocities[:-1], velocities[1:]) if v2 not 0]


def vcr(v1: float, v2: float) -> float:
    """Calculate Velocity Change Rate between two velocities
    Args:
        v1 (float): Velocity 1
        v2 (float): Velocity 2
    Returns:
        float: Velocity Change Rate between the two inputs
    """
    return (v2 - v1) / v2
