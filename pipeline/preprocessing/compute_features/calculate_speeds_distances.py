# append root to path
import sys
from os.path import dirname, realpath

current_dir = dirname(realpath(__file__))
parentdir = dirname(dirname(dirname(current_dir)))
sys.path.append(parentdir)

import json
from math import sqrt
import pandas as pd # type: ignore
from pipeline.preprocessing.compute_features.type_alias import Trip

coordinate = tuple[float, float]  # type alias


def calc_utm_dist(utm1: coordinate, utm2: coordinate) -> float:
    """calculate the distance between two utm coordinates
    uses formula for euclidian distance, which is not 100% accurate?

    Args:
        utm1 (coordinate): a tuple of x and y coordinate
        utm2 (coordinate): a tuple of x and y coordinate

    Returns:
        float: distance between the two coordinates in meters
        (utm is already in meters, so no conversion)
    """
    x1, y1 = utm1
    x2, y2 = utm2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def ms_to_kmh(speeds: list[float]) -> list[float]:
    """convert list of speeds in meters per second to list of speeds with
    km per second.

    Args:
        speeds (list[float]): list of speeds

    Returns:
        list[float]: new list of speeds
    """
    return [speed * 3.6 for speed in speeds]


def calculate_distances(trip: Trip) -> list[float]:
    """Calculate the distances between the coordinates in a trip.

    Args:
        trip (Trip): a list of coordinates with time, ie. a trip

    Returns:
        list[float]: a list of distances in meters
    """
    # create two lists of the coordinates, list without last element and list without first element
    coordinates_with_following_coordinates = zip(trip[:-1], trip[1:])
    return [
        calc_utm_dist((x, y), (sx, sy))
        for (x, y, _), (sx, sy, _) in coordinates_with_following_coordinates
    ]


def calculate_speeds(trip: Trip, distances: list[float]) -> list[float]:
    """Calculates the speeds in km/h betweeen each CoordinateWithTime in the trip.

    Args:
        trip (Trip): list coordinates with time

    Returns:
        list[float]: list of speeds in km/h
    """
    time_differences = [t2 - t1 for (_, _, t1), (_, _, t2) in zip(trip[:-1], trip[1:])]
    speed_ms = [
        distance / scaler for distance, scaler in zip(distances, time_differences)
    ]
    return ms_to_kmh(speed_ms)
