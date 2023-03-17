
myList = [1,2,3,4]
def wrapper_VCR(velocities: list[float]) -> list[float]:
    return map(lambda v1, v2: VCR(v1, v2), velocities)


def VCR(v1: float, v2: float) -> float:
    """
    :param v1: Velocity 1
    :type v1: float
    :param v2: Velocity 2
    :type v2: float
    :return: The Velocity Change Rate between the input velocities
    :rtype: float
    """
    return (v2 - v1)/v2


def main() :
    print(wrapper_VCR(myList))
    #TODO: Figure out data format
    #TODO: Use existing velocity calculation for generation velocities
    #TODO: Pr. trip calculate VCR

if __name__ == "__main__":

    main()