import numpy as np
from sklearn.compose import make_column_transformer

from pipeline.models.utils.scoring import SPEED_LIMITS
from pipeline.preprocessing.compute_features.feature import Feature, FeatureList
from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


class SKFormatter:
    """Used to format our dataset, such that it can be used for sklearn models."""

    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.2,
        discard_features: FeatureList = None,
        target: Feature = None
    ):
        """Init

        Args:
            dataset_path (str): path to pickle dataset
            target_feature (Feature): The future to be used as target. Defaults to Feature.HAST_GAELDENDE_HAST.
            test_size (float, optional): amount of the dataset to be used to testing. Defaults to 0.2.
        """
        self.df = pd.read_pickle(dataset_path).head(1000)
        self.test_size = test_size

        if discard_features is None:  # bad practice to use objects as default params
            self.discard_features = FeatureList(
                [
                    Feature.OSM_ID,
                    Feature.COORDINATES,
                    Feature.CPR_VEJNAVN,
                    Feature.HAST_SENEST_RETTET,
                    Feature.DISTANCES,
                ]
            )
        else:
            self.discard_features = discard_features

        if target is None:
            self.target_feature = Feature.HAST_GAELDENDE_HAST.value
        else:
            self.target_feature = target.value

    def generate_train_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate the test train split.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """
        # don't train with the following features
        self.df = self.df.drop(self.discard_features, axis=1)

        # extract and encode target
        self.df = self.df.rename(
            columns={self.target_feature: Feature.TARGET.value}
        )
        # self.__encode_target()
        y = self.df[Feature.TARGET.value].values
        self.df = self.df.drop([Feature.TARGET.value], axis=1)

        # encode features
        self.__encode_categorical_features()
        self.__encode_single_value_features()

        # generate features np array
        x = self.__generate_x()

        # x_train, x_test, y_train, y_test
        return self.__test_train_split(x, y)

    def __generate_x(self) -> np.ndarray:
        """Create x ie. numpy array the features, without target.

        Returns:
            np.ndarray: numpy ndarray of the features without target.
        """

        # flatten 2d arrays to 1d
        for feature in Feature.array_2d_features() - self.discard_features:
            self.df[feature] = self.df[feature].apply(lambda row: sum(row, []))

        for feature in Feature.array_features() - self.discard_features:
            # remove nones that might occur in the arrays
            self.df[feature] = self.df[feature].apply(
                lambda arr: [elem for elem in arr if arr is not None]
            )
            # pad the arrays, so they are all the same length, necessary for sklearn
            self.df[feature] = pad_sequences(self.df[feature], padding="post").tolist()
            # Make each array a numpy array
            self.df[feature] = self.df[feature].apply(lambda arr: np.array(arr))

        # Combine all the features into a numpy array
        xs = [self.df[f].values.tolist() for f in self.df.columns]
        x = np.concatenate(xs, axis=1)

        # replace all nan values with 0
        np.nan_to_num(x, 0)

        return x

    def __encode_single_value_features(self) -> None:
        """
        Encode the non array features. They must be numpy arrays.
        """
        to_encode = Feature.array_features().not_in(self.df.columns)
        for f in to_encode:
            self.df[f] = self.df[f].apply(lambda val: np.array([val]))

    def __encode_categorical_features(self) -> None:
        """
        One-hot encode categorical features.
        """
        categorical_features = Feature.categorical_features() - self.discard_features
        one_hot_encoded = pd.get_dummies(self.df[categorical_features], dtype=int)
        self.df = pd.concat([one_hot_encoded, self.df], axis=1)
        self.df = self.df.drop(categorical_features, axis=1)

    # Don't think these methods are necessary, but mabye they are...

    # def __encode_target(self) -> None:
    #     self.df[Feature.TARGET.value] = self.encode_speed_limits(self.df[Feature.TARGET.value])
    #     print(self.df[Feature.TARGET.value].unique())
    #
    # def encode_speed_limits(self, speed_limits: np.ndarray) -> np.ndarray:
    #     self.mapping = {limit: index for index, limit in enumerate(np.unique(speed_limits))}
    #     return [self.mapping[limit] for limit in speed_limits]
    #
    # def decode_speed_limits(self, encoded_speed_limits: np.ndarray) -> np.ndarray:
    #     mapping = {index: limit for limit, index in self.mapping}
    #     return [mapping[limit] for limit in encoded_speed_limits]

    def __test_train_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split x and y into a train and test sets.
        The dataset will always be split, such that the first
        test_size number of rows are used for testing, and the
        remaining are used for training. This is useful for stitching
        the predictions on the test set back to the original osm_ids.

        Args:
            x (np.ndarray): numpy array with features
            y (np.ndarray): numpy array with target

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """

        # return train_test_split(x, y, test_size=self.test_size, random_state=42)

        # Calculate the index for splitting the dataset
        split_idx = int(len(x) * (1 - self.test_size))

        # Split the dataset into training and testing sets
        x_test, y_test = x[:split_idx], y[:split_idx]
        x_train, y_train = x[split_idx:], y[split_idx:]

        return x_train, x_test, y_train, y_test
