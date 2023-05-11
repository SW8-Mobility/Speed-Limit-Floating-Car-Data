from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.preprocessing.compute_features.feature import Feature, FeatureList
from tests.mock_dataset import mock_dataset
import pandas as pd
import pytest
import numpy as np


# def test_encode_single_value_features():
#     df = mock_dataset(10, 3).drop([Feature.OSM_ID.value], axis=1)
#     skf = SKFormatter(df)

#     # accessing private method by name mangling
#     skf._SKFormatter__encode_single_value_features()  # type: ignore

#     # All the features, that are not array features, ie. single value features
#     single_value_features = Feature.array_features().not_in(skf.df.columns)  # type: ignore

#     for f in single_value_features:
#         assert isinstance(skf.df[f][0], np.ndarray)


def test_encode_categorical_features_columns_exist():
    row_num = 10
    df = mock_dataset(row_num, 3).drop([Feature.OSM_ID.value], axis=1)
    skf = SKFormatter(df)

    # accessing private method by name mangling
    skf._SKFormatter__encode_categorical_features()  # type: ignore

    expected_one_hot_cols = [
        f"{Feature.VEJSTIKLASSE.value}_{i}" for i in range(row_num)
    ] + [
        f"{Feature.VEJTYPESKILTET.value}_motorvej",
        f"{Feature.VEJTYPESKILTET.value}_byvej",
    ]

    for e in expected_one_hot_cols:
        assert e in skf.df.columns


def test_encode_categorical_features_are_0_or_1():
    row_num = 10
    df = mock_dataset(row_num, 3).drop([Feature.OSM_ID.value], axis=1)
    skf = SKFormatter(df)

    # accessing private method by name mangling
    skf._SKFormatter__encode_categorical_features()  # type: ignore

    one_hot_cols = [f"{Feature.VEJSTIKLASSE.value}_{i}" for i in range(row_num)] + [
        f"{Feature.VEJTYPESKILTET.value}_motorvej",
        f"{Feature.VEJTYPESKILTET.value}_byvej",
    ]

    for c in one_hot_cols:
        actual = list(skf.df[c].unique())  # unique values of a one hot columns
        assert all(val in [0, 1] for val in actual)  # all values must be either 1 or 0


def test_generate_train_test_split_all_numbers_despite_nones():
    df = mock_dataset(10, 3)
    df.loc[0, Feature.MINS.value][0] = None  # type: ignore
    df.loc[0, Feature.AGGREGATE_MIN.value] = None
    df.loc[0, Feature.HAST_GAELDENDE_HAST.value] = None
    skf = SKFormatter(df)
    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    assert not x_train.isnull().values.any()
    assert not x_test.isnull().values.any()
    assert not y_train.isnull().values.any()
    assert not y_test.isnull().values.any()


def test_generate_train_test_split_only_ints_or_floats():  # TODO: there should be no arrays
    df = mock_dataset(10, 3)
    df.loc[0, Feature.MINS.value].pop(0)  # make one array inconsistent length
    skf = SKFormatter(df)

    x_train, x_test, y_train, y_test = skf.generate_train_test_split()
    for t in x_train.dtypes:
        assert t in [np.float64, np.int64, np.int32, np.float32]

    for t in x_test.dtypes:
        assert t in [np.float64, np.int64, np.int32, np.float32]

    assert y_train.dtype == np.int64
    assert y_test.dtype == np.int64


def test_generate_train_test_split_splits_are_correct_lengths():
    df = mock_dataset(10, 3)
    test_size_20_percent = 0.2
    skf = SKFormatter(df, test_size=test_size_20_percent)

    df = mock_dataset(10, 3)
    skf = SKFormatter(df)

    train_expected_length = 8
    test_expected_length = 2
    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    assert len(x_train) == train_expected_length
    assert len(y_train) == train_expected_length

    assert len(x_test) == test_expected_length
    assert len(y_test) == test_expected_length


@pytest.mark.parametrize(
    "row_num, train_expected_row_num, test_expected_row_num",
    [
        (10, 8, 2),
        (20, 16, 4),
        (100, 80, 20),
        (1000, 800, 200),
    ],
)
def test_generate_train_test_split_splits_are_correct_shape(
    row_num, train_expected_row_num, test_expected_row_num
):
    df = mock_dataset(row_num, 3)
    discard_features = (
        FeatureList(
            [
                Feature.OSM_ID,
                Feature.COORDINATES,
                Feature.CPR_VEJNAVN,
                Feature.HAST_SENEST_RETTET,
                Feature.DISTANCES,
            ]
        )
        + Feature.categorical_features()
    )
    skf = SKFormatter(
        df, full_dataset=True, test_size=0.2, discard_features=discard_features
    )
    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    # check test and train have correct number of rows
    assert len(x_train) == train_expected_row_num
    assert len(y_train) == train_expected_row_num

    assert len(x_test) == test_expected_row_num
    assert len(y_test) == test_expected_row_num

    # assert only 1d array
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # assert there are correct number of columns
    # print(x_train.columns)
    col_num_without_categorical_features = 145
    assert len(x_train.columns) == col_num_without_categorical_features
    assert len(x_test.columns) == col_num_without_categorical_features


def test_generate_train_test_split_splits_no_duplicates_despite_duplicates_in_dataset():
    row_num = 10
    expected_row_num = row_num
    df = mock_dataset(row_num, 3)

    # duplicate a row, and add it
    duplicated_row = df.iloc[0].copy()
    df.loc[len(df)] = duplicated_row  # type: ignore

    skf = SKFormatter(df)
    x_train, x_test, _, _ = skf.generate_train_test_split()
    actual_row_num = len(x_train) + len(x_test)

    assert actual_row_num == expected_row_num


def test_get_params():
    test_size = 0.2
    discard_features = FeatureList(
        [
            Feature.OSM_ID,
            Feature.COORDINATES,
            Feature.CPR_VEJNAVN,
            Feature.HAST_SENEST_RETTET,
            Feature.DISTANCES,
        ]
    )
    target_feature = Feature.HAST_GAELDENDE_HAST
    df = mock_dataset()

    skf = SKFormatter(df)

    expected_params = [
        ("full_dataset", False),
        ("dataset_size", 1000),
        ("test_size", test_size),
        ("discard_features", discard_features),
        ("target_feature", target_feature.value),
        ("input_df_columns", list(df.columns)),
    ]

    actual_params = skf.params

    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_params, actual_params.items()
    ):
        assert expected_name == actual_name
        assert expected_value == actual_value


def test_get_params_after_generating_test_train_split():
    test_size = 0.2
    discard_features = FeatureList(
        [
            Feature.OSM_ID,
            Feature.COORDINATES,
            Feature.CPR_VEJNAVN,
            Feature.HAST_SENEST_RETTET,
            Feature.DISTANCES,
        ]
    )
    target_feature = Feature.HAST_GAELDENDE_HAST
    df = mock_dataset(10, 3)

    skf = SKFormatter(df)

    new_col_num = 20
    vejstiklasse = [f"vejstiklasse_{i}" for i in range(10)]
    vejtypeskiltet = ["vejtypeskiltet_byvej", "vejtypeskiltet_motorvej"]
    speeds = [f"speeds_{i}" for i in range(new_col_num)]
    means = [f"means_{i}" for i in range(new_col_num)]
    aggregate_mean = ["aggregate_mean"]
    mins = [f"mins_{i}" for i in range(new_col_num)]
    aggregate_min = ["aggregate_min"]
    maxs = [f"maxs_{i}" for i in range(new_col_num)]
    aggregate_max = ["aggregate_max"]
    rolling_averages = [f"rolling_averages_{i}" for i in range(new_col_num)]
    medians = [f"medians_{i}" for i in range(new_col_num)]
    aggregate_median = ["aggregate_median"]
    vcr = [f"vcr_{i}" for i in range(new_col_num)]
    hast_generel_hast = ["hast_generel_hast"]
    hast_gaeldende_hast = ["hast_gaeldende_hast"]

    processed_df_columns = [
        *vejstiklasse,
        *vejtypeskiltet,
        *aggregate_mean,
        *aggregate_min,
        *aggregate_max,
        *aggregate_median,
        *hast_generel_hast,
        *hast_gaeldende_hast,
        *means,
        *mins,
        *maxs,
        *medians,
        *speeds,
        *rolling_averages,
        *vcr,
    ]

    expected_params = [
        ("full_dataset", False),
        ("dataset_size", 1000),
        ("test_size", test_size),
        ("discard_features", discard_features),
        ("target_feature", target_feature.value),
        ("input_df_columns", list(df.columns)),
        ("processed_df_columns", processed_df_columns),
    ]

    skf.generate_train_test_split()
    actual_params = skf.params
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_params, actual_params.items()
    ):
        assert expected_name == actual_name
        assert expected_value == actual_value
