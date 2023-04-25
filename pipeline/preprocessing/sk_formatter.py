import numpy as np
from sklearn.compose import make_column_transformer

from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


class SKFormatter:
    """Used to format our dataset, such that it can be used for sklearn models."""

    def __init__(
        self,
        dataset_path: str,
        target_feature: Feature = Feature.HAST_GAELDENDE_HAST,
        test_size: float = 0.2,
    ):
        """Init

        Args:
            dataset_path (str): path to pickle dataset
            target_feature (Feature): The future to be used as target. Defaults to Feature.HAST_GAELDENDE_HAST.
            test_size (float, optional): amount of the dataset to be used to testing. Defaults to 0.2.
        """
        self.df = pd.read_pickle(dataset_path)
        self.target = target_feature.value
        self.discard_features = [Feature.OSM_ID.value]
        self.test_size = test_size

    def generate_train_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate the test train split.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """
        # encode features
        self.__encode_categorical_features()
        self.__encode_single_value_features()

        # extract target
        self.df = self.df.rename(
            columns={Feature.HAST_GAELDENDE_HAST.value: Feature.TARGET.value}
        )
        y = self.df[Feature.TARGET.value].values
        self.df = self.df.drop([Feature.TARGET.value], axis=1)

        # generate features np array
        x = self.__generate_x()

        # x_train, x_test, y_train, y_test
        return self.__test_train_split(x, y)

    def __generate_x(self) -> np.ndarray:
        """Create x ie. numpy array the features, without target.

        Returns:
            np.ndarray: numpy ndarray of the features without target.
        """
        # don't train with the following features
        self.df = self.df.drop(self.discard_features, axis=1)

        # make 2d array features 1d by flattening them
        for feature in Feature.array_2d_features():
            self.df[feature] = self.df[feature].apply(lambda row: sum(row, []))

        for feature in Feature.array_1d_features() + Feature.array_2d_features():
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

        return x

    def __encode_single_value_features(self) -> None:
        """
        Encode the numeric value features. They must be numpy arrays.
        """
        for f in Feature.numeric_features():
            self.df[f] = self.df[f].apply(lambda val: np.array([val]))

    def __encode_categorical_features(self) -> None:
        """
        One-hot encode categorical values.
        """
        transformer = make_column_transformer(
            (OneHotEncoder(), Feature.categorical_features()), remainder="passthrough"
        )

        transformed = transformer.fit_transform(self.df)
        self.df = pd.DataFrame(transformed, columns=transformer.get_feature_names())

    def __test_train_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split x and y into a train and test sets.

        Args:
            x (np.ndarray): numpy array with features
            y (np.ndarray): numpy array with target

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """
        return train_test_split(  # type: ignore
            x, y, test_size=self.test_size, random_state=42
        )
