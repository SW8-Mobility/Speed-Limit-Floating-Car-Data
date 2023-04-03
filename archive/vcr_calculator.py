class TooFewElementsException(Exception):
    """Raised when there are too few elements in the input list"""

    pass


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
    return list(map(lambda v1, v2: vcr(v1, v2), velocities[:-1], velocities[1:]))


def vcr(v1: float, v2: float) -> float:
    """Calculate Velocity Change Rate between two velocities
    Args:
        v1 (float): Velocity 1
        v2 (float): Velocity 2
    Returns:
        float: Velocity Change Rate between the two inputs
    """
    return (v2 - v1) / v2
