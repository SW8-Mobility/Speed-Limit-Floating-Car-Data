import pytest
import pandas as pd  # type: ignore
from data_exploration.speed_graph import (
    select_osm_rows,
    flatten_and_concat_speeds,
    floor_list,
)


@pytest.mark.parametrize(
    "input_value, index_list, expected_value",
    [
        (
            {"osm_id": [2080, 2081], "speeds": [[[10, 20]], [[30], [40]]]},
            [2080, 2081],
            {"osm_id": [2080, 2081], "speeds": [[[10, 20]], [[30], [40]]]},
        ),
        (
            {
                "osm_id": [2080, 2081],
                "speeds": [
                    [[10, 11], [11, 12]],
                    [[10, 11], [11, 12]],
                ],  # the actual data is on this format
            },
            [2082],
            {"osm_id": [2082], "speeds": [[[10, 11], [11, 12]]]},
        ),
    ],
)
def test_selected_speeds(input_value, index_list, expected_value):
    input_df = pd.DataFrame(data=input_value)
    expected_df = pd.DataFrame(data=expected_value)

    actual_df = select_osm_rows(input_df, index_list)

    actual_df.equals(expected_df)


@pytest.mark.parametrize(
    "input_value, expected_value",
    [
        (
            {
                "osm_id": [2080, 2081],
                "speeds": [[[10, 11], [11, 12]], [[10, 11], [11, 12]]],
            },
            [10, 11, 11, 12, 10, 11, 11, 12],
        ),
        (
            {"osm_id": [2080], "speeds": [[[10, 11], [11, 12]]]},
            [10, 11, 11, 12],
        ),
    ],
)
def test_flatten_and_concat_speed_list(input_value, expected_value):
    input_df = pd.DataFrame(data=input_value)

    actual_value = flatten_and_concat_speeds(input_df)

    assert actual_value == expected_value


@pytest.mark.parametrize(
    "input_value, expected_value", [([5.5, 10.2, 15.7], [5, 10, 15])]
)
def test_floor_list(input_value, expected_value):
    actual_value = floor_list(input_value)

    assert actual_value == expected_value
