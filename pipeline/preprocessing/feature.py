from enum import Enum


class Feature(Enum):
    ID = "id"
    COORDINATES = "coordinates"
    MEANS = "means"
    AGGREGATE_MEAN = "aggregate_mean"
    MAXS = "maxs"
    AGGREGATE_MAX = "aggregate_max"
    ROAD_TYPE = "road_type"
    ROLLING_AVERAGES = "rolling_averages"
    LEVEL = "level"
    MEDIANS = "medians"
    AGGREGATE_MEDIANS = "aggregate_medians"
    VCR = "vcr"
    DAY_OF_WEEK = "day_of_week"
    TIME_GROUP = "time_group"
    # more to come

    def __str__(self):
        self.value

    def type_check(self):
        """Might be useful to type check the format of a feature later.
        example: maxs should be a list of floats
        """
        pass
