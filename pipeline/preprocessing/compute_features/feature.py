from enum import Enum


# enum to represent our features
class Feature(Enum):
    ID = "id"
    COORDINATES = "coordinates"
    SPEEDS = "speeds"
    DISTANCES = "distances"
    MEANS = "means"
    AGGREGATE_MEAN = "aggregate_mean"
    MINS = "mins"
    AGGREGATE_MIN = "aggregate_min"
    MAXS = "maxs"
    AGGREGATE_MAX = "aggregate_max"
    ROAD_TYPE = "road_type"
    ROLLING_AVERAGES = "rolling_averages"
    LEVEL = "level"
    MEDIANS = "medians"
    AGGREGATE_MEDIAN = "aggregate_median"
    VCR = "vcr"
    DAY_OF_WEEK = "day_of_week"
    TIME_GROUP = "time_group"
    # more to come

    def __str__(self) -> str:
        return self.value
