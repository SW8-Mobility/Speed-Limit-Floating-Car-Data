import pytest

from pipeline.preprocessing.compute_features.feature import Feature, FeatureList


@pytest.mark.parametrize(
    "feature1, feature2, expected",
    [
        (Feature.SPEEDS, Feature.SPEEDS, True),
        (Feature.SPEEDS, Feature.DISTANCES, False),
    ],
)
def test_feature_eq(feature1, feature2, expected):
    actual = feature1 == feature2
    assert actual == expected


@pytest.mark.parametrize(
    "feature_list1, feature_list2, expected",
    [
        (FeatureList([Feature.SPEEDS]), FeatureList([Feature.SPEEDS]), True),
        (FeatureList([Feature.SPEEDS]), FeatureList([Feature.DISTANCES]), False),
        (FeatureList([]), FeatureList([]), True),
        (
            FeatureList([Feature.DISTANCES, Feature.SPEEDS]),
            FeatureList([Feature.DISTANCES]),
            False,
        ),
        (FeatureList([Feature.DISTANCES, Feature.SPEEDS]), FeatureList([]), False),
    ],
)
def test_feature_list_eq(feature_list1, feature_list2, expected):
    actual = feature_list1 == feature_list2
    assert actual == expected


@pytest.mark.parametrize(
    "input_list1, input_list2, expected",
    [
        ([Feature.SPEEDS], [Feature.SPEEDS.value], []),
        (
            [Feature.SPEEDS],
            [Feature.SPEEDS.value, "onehot_encoded_feature"],
            ["onehot_encoded_feature"],
        ),
        ([], [Feature.SPEEDS.value], [Feature.SPEEDS.value]),
    ],
)
def test_feature_list_not_in(input_list1, input_list2, expected):
    fl = FeatureList(input_list1)
    actual = fl.not_in(input_list2)
    assert actual == expected


@pytest.mark.parametrize(
    "input_list1, input_list2, expected_list",
    [
        ([Feature.SPEEDS], [Feature.SPEEDS], []),
        ([Feature.SPEEDS, Feature.DISTANCES], [Feature.SPEEDS], [Feature.DISTANCES]),
        ([], [Feature.SPEEDS], []),
    ],
)
def test_feature_list_minus(input_list1, input_list2, expected_list):
    expected = FeatureList(expected_list)
    fl1 = FeatureList(input_list1)
    fl2 = FeatureList(input_list2)
    actual = fl1 - fl2
    assert actual == expected


@pytest.mark.parametrize(
    "feature_list1, feature_list2, expected",
    [
        (
            FeatureList([Feature.SPEEDS]),
            FeatureList([Feature.SPEEDS]),
            FeatureList([Feature.SPEEDS, Feature.SPEEDS]),
        ),
        (
            FeatureList([Feature.SPEEDS]),
            FeatureList([Feature.DISTANCES]),
            FeatureList([Feature.SPEEDS, Feature.DISTANCES]),
        ),
        (
            FeatureList([Feature.SPEEDS, Feature.DISTANCES]),
            FeatureList([]),
            FeatureList([Feature.SPEEDS, Feature.DISTANCES]),
        ),
        (
            FeatureList([]),
            FeatureList([Feature.SPEEDS, Feature.DISTANCES]),
            FeatureList([Feature.SPEEDS, Feature.DISTANCES]),
        ),
    ],
)
def test_feature_list_add(feature_list1, feature_list2, expected):
    actual = feature_list1 + feature_list2
    assert actual == expected


@pytest.mark.parametrize(
    "feature_list, expected_list",
    [
        (
            [Feature.SPEEDS, Feature.DISTANCES, Feature.AGGREGATE_MAX],
            [
                Feature.SPEEDS.value,
                Feature.DISTANCES.value,
                Feature.AGGREGATE_MAX.value,
            ],
        ),
        ([], []),
        ([Feature.VEJSTIKLASSE], [Feature.VEJSTIKLASSE.value]),
    ],
)
def test_feature_list_iter(feature_list, expected_list):
    fl = FeatureList(feature_list)
    for feature, expected in zip(fl, expected_list):
        assert feature == expected


def test_feature_array_1d_features():
    expected = FeatureList(
        [
            Feature.MEANS,
            Feature.MINS,
            Feature.MAXS,
            Feature.MEDIANS,
        ]
    )

    actual = Feature.array_1d_features()

    assert actual == expected


def test_feature_array_2d_features():
    expected = FeatureList(
        [
            Feature.COORDINATES,
            Feature.DISTANCES,
            Feature.SPEEDS,
            Feature.ROLLING_AVERAGES,
            Feature.VCR,
        ]
    )

    actual = Feature.array_2d_features()

    assert actual == expected


def test_feature_categorical_features():
    expected = FeatureList(
        [
            Feature.VEJSTIKLASSE,
            Feature.VEJTYPESKILTET,
        ]
    )

    actual = Feature.categorical_features()

    assert actual == expected


def test_feature_array_features():
    expected = FeatureList(
        [
            Feature.MEANS,
            Feature.MINS,
            Feature.MAXS,
            Feature.MEDIANS,
            Feature.COORDINATES,
            Feature.DISTANCES,
            Feature.SPEEDS,
            Feature.ROLLING_AVERAGES,
            Feature.VCR,
        ]
    )

    actual = Feature.array_features()

    assert actual == expected
