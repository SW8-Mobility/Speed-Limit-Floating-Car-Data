from enum import Enum

# enum to represent our features
class Feature(Enum):
    OSM_ID = "osm_id"
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
    SPEED_LIMIT_TARGET = "speed_limit_target"
    SPEED_LIMIT_PREDICTED = "speed_limit_predicted"

    # vejman features
    CPR_VEJNAVN = 'cpr_vejnavn'
    HAST_GENEREL_HAST = 'hast_generel_hast'
    HAST_GAELDENDE_HAST = 'hast_gaeldende_hast'
    VEJSTIKLASSE = 'vejstiklasse'
    VEJTYPESKILTET = 'vejtypeskiltet'
    HAST_SENEST_RETTET = "hast_senest_rettet"
    TARGET = "target"

    @staticmethod
    def array_1d_features():
        """
        returns a list of the names of features where the type is 1d-arrays.
        """
        return [
            Feature.MEANS.value,
            Feature.MINS.value,
            Feature.MAXS.value,
            Feature.MEDIANS.value
        ]

    @staticmethod
    def array_2d_features() -> list[str]:
        """
        returns a list of the names of features where the type is 2d arrays.
        """
        return [
            Feature.DISTANCES.value,
            Feature.SPEEDS.value,
            Feature.ROLLING_AVERAGES.value,
            Feature.VCR.value
        ]

    @staticmethod
    def numeric_features() -> list[str]:
        """
        returns a list of the names of features where the type is a int/float.
        """
        return [
            Feature.AGGREGATE_MEAN.value,
            Feature.AGGREGATE_MAX.value,
            Feature.AGGREGATE_MEDIAN.value,
            Feature.AGGREGATE_MIN.value,
            Feature.HAST_GENEREL_HAST.value,
        ]

    @staticmethod
    def categorical_features() -> list[str]:
        """
        returns a list of the names of features where the feature is categorical.
        """
        return [
            Feature.COORDINATES.value,
            Feature.CPR_VEJNAVN.value,
            Feature.HAST_SENEST_RETTET.value,
            Feature.VEJSTIKLASSE.value,
            Feature.VEJTYPESKILTET.value,
        ]

    def __str__(self) -> str:
        return self.value