from typing import Callable
import pandas as pd
from pipeline.preprocessing.compute_features.feature import Feature
from pipeline.models.utils.scoring import SPEED_LIMITS
import itertools


def mock_speed_limits(count: int) -> list[int]:
    return list(itertools.islice(itertools.cycle(SPEED_LIMITS), count))


def mock_road_type(count: int) -> list[str]:
    return list(itertools.islice(itertools.cycle(["motorvej", "byvej"]), count))


def mock_col(trips: int, rows: int) -> Callable:
    def list_comp(value) -> list:
        if trips > 1:
            return [[value for _ in range(trips)] for _ in range(rows)]
        else:
            return [value for _ in range(rows)]

    return list_comp


def mock_dataset(row_num: int = 10, trip_num: int = 3) -> pd.DataFrame:
    col_mocker = mock_col(trip_num, row_num)
    agg_col_mocker = mock_col(1, row_num)
    data = {
        Feature.OSM_ID.value: list(range(row_num)),
        Feature.COORDINATES.value: [
            [[[100.0, 100 + (10 * i), 1000000000.0 + i] for i in range(trip_num)]]
            for _ in range(row_num)
        ],
        Feature.DISTANCES.value: col_mocker([10, 20, 30]),
        Feature.SPEEDS.value: col_mocker([36, 72, 108]),
        Feature.MEANS.value: col_mocker(70),
        Feature.AGGREGATE_MEAN.value: agg_col_mocker(70),
        Feature.MEANS.value: col_mocker(36),
        Feature.AGGREGATE_MEAN.value: agg_col_mocker(36),
        Feature.MAXS.value: col_mocker(108),
        Feature.AGGREGATE_MAX.value: agg_col_mocker(108),
        Feature.ROLLING_AVERAGES.value: col_mocker([70 for _ in range(trip_num - 1)]),
        Feature.MEDIANS.value: col_mocker(72),
        Feature.AGGREGATE_MEDIAN.value: agg_col_mocker(72),
        Feature.VCR.value: col_mocker([70 for _ in range(trip_num - 1)]),
        Feature.CPR_VEJNAVN.value: [f"street_{i}" for i in range(row_num)],
        Feature.HAST_GENEREL_HAST.value: mock_speed_limits(row_num),
        Feature.HAST_GAELDENDE_HAST.value: mock_speed_limits(row_num),
        Feature.VEJSTIKLASSE.value: [str(i) for i in range(row_num)],
        Feature.VEJTYPESKILTET.value: mock_road_type(row_num),
        Feature.HAST_SENEST_RETTET.value: agg_col_mocker("20/07/2014"),
    }

    return pd.DataFrame(data=data)
