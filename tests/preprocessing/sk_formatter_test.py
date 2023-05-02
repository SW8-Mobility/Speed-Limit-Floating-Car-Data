from pipeline.preprocessing.sk_formatter import SKFormatter
from pipeline.preprocessing.compute_features.feature import Feature
from tests.mock_dataset import mock_dataset
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
        assert all(val in [0,1] for val in actual)  # all values must be either 1 or 0

def test_generate_train_test_split_all_numbers():
    df = mock_dataset(10, 3)
    df[Feature.AGGREGATE_MIN.value].iloc[0] = None
    df[Feature.MINS.value].iloc[0][0] = None
    skf = SKFormatter(df)

    # accessing private method by name mangling
    x_train, x_test, y_train, y_test = skf.generate_train_test_split()  # type: ignore

    assert np.char.isnumeric(x_train.astype(str)).all()
    assert np.char.isnumeric(x_test.astype(str)).all()
    assert np.char.isnumeric(y_train.astype(str)).all()
    assert np.char.isnumeric(y_test.astype(str)).all()
