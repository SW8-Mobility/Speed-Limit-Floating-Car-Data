def map_vcr(velocities: list[float]) -> list[float]:
    return list(map(lambda v1, v2: vcr(v1, v2), velocities[:-1], velocities[1:]))


def vcr(v1: float, v2: float) -> float:
    """
    :param v1: Velocity 1
    :type v1: float
    :param v2: Velocity 2
    :type v2: float
    :return: The Velocity Change Rate between the input velocities
    :rtype: float
    """
    return (v2 - v1)/v2


def main():
    print(map_vcr([1, 2, 3, 4]))


if __name__ == "__main__":
    main()



