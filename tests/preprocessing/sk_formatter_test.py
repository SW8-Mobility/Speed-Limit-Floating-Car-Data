from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.preprocessing.compute_features.feature import Feature
from tests.mock_dataset import mock_dataset
import pandas as pd
import pytest
import numpy as np


def test_encode_single_value_features():
    df = mock_dataset(10, 3).drop([Feature.OSM_ID.value], axis=1)
    skf = SKFormatter(df)

    # accessing private method by name mangling
    skf._SKFormatter__encode_single_value_features()  # type: ignore

    # All the features, that are not array features, ie. single value features
    single_value_features = Feature.array_features().not_in(skf.df.columns)  # type: ignore

    for f in single_value_features:
        assert isinstance(skf.df[f][0], np.ndarray)


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
    skf = SKFormatter(df)

    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    assert np.isfinite(x_train).all()
    assert np.isfinite(x_test).all()
    assert np.isfinite(y_train).all()
    assert np.isfinite(y_test).all()

def test_generate_train_test_split_arrays_in_cols_are_same_length():
    df = mock_dataset(10, 3)
    df.loc[0, Feature.MINS.value].pop(0)  # make one array inconsistent length

    skf = SKFormatter(df)

    try:
        skf.generate_train_test_split()  
        assert True
    except ValueError:
        pytest.fail(
            "ValueError: all the input arrays must have same number of dimensions.\nPadding is not working."
        )


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
    
def test_generate_train_test_split_splits_are_correct_shape():
    df = mock_dataset(10, 3)
    skf = SKFormatter(df)

    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    # check x and y have same number of rows
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    
    # assert only 1d array
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1

def test_generate_train_test_split_splits_no_duplicates_despite_duplicates_in_dataset():
    row_num = 10
    expected_row_num = row_num
    df = mock_dataset(row_num, 3)

    # duplicate a row, and add it
    duplicated_row = df.iloc[0].copy()
    df.loc[len(df)] = duplicated_row # type: ignore

    skf = SKFormatter(df)
    x_train, x_test, _, _ = skf.generate_train_test_split()
    actual_row_num = len(x_train) + len(x_test)

    assert actual_row_num == expected_row_num 
