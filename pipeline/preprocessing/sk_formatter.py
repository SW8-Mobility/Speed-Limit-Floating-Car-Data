from typing import Union
import numpy as np
from pipeline.preprocessing.compute_features.feature import Feature, FeatureList
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences  # type: ignore


class SKFormatter:
    """Used to format our dataset, such that it can be used for sklearn models."""

    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        test_size: float = 0.2,
        discard_features: FeatureList = None,  # type: ignore
        target: Feature = None,  # type: ignore
        dataset_size: int = 1000,
        full_dataset: bool = False,
    ):
        """init

        Args:
            dataset_path (str or pd.DataFrame): path to pickle dataset or dataset as dataframe.
            test_size (float, optional): amount of the dataset to be used to testing. Defaults to 0.2.
            discard_features (FeatureList, optional): Optional list of features to discard. Defaults to None.
            target (Feature, optional): The feature to be used as target. Defaults to None.
            dataset_size (int, optional): only use n number of rows. Defaults to 1000.
            full_dataset (bool, optional): Use full dataset or not. Defaults to False.
        """
        self.df = pd.read_pickle(dataset) if isinstance(dataset, str) else dataset

        if not full_dataset:
            self.df = self.df.head(dataset_size)

        self.test_size = test_size

        self.discard_features: FeatureList = (
            FeatureList(
                [
                    Feature.OSM_ID,
                    Feature.COORDINATES,
                    Feature.CPR_VEJNAVN,
                    Feature.HAST_SENEST_RETTET,
                    Feature.DISTANCES,
                ]
            )
            if discard_features is None
            else discard_features
        )

        self.target_feature: str = (
            Feature.HAST_GAELDENDE_HAST.value if target is None else target.value
        )

        self.params = self.__params()

    def __params(self) -> dict:
        params = self.__dict__.copy()
        df = params.pop("df")
        params["input_df_columns"] = list(df.columns)
        return params

    def generate_train_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate the test train split.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """
        # don't train with the following features
        self.df = self.df.drop(self.discard_features, axis=1)

        self.__remove_duplicates()

        # extract target
        y = self.__generate_y()

        # encode features
        self.__encode_categorical_features()
        self.__encode_single_value_features()
        self.__encode_array_features()

        # save df config after processing
        self.params["processed_df_columns"] = list(self.df.columns)

        # generate features np array
        x = self.__generate_x()

        # x_train, x_test, y_train, y_test
        return self.__test_train_split(x, y)

    def __remove_duplicates(self) -> None:
        """Remove duplicates from dataset.
        should ideally have been done before.
        """
        cols = self.df.columns
        self.df = self.df[cols].loc[self.df[cols].astype(str).drop_duplicates().index]

    def __generate_y(self) -> np.ndarray:
        """Extract target from dataframe as y, and remove it
        from the df.

        Returns:
            np.ndarray: a numpy array of target values.
        """
        self.df = self.df.rename(columns={self.target_feature: Feature.TARGET.value})
        y = self.df[Feature.TARGET.value].values
        self.df = self.df.drop([Feature.TARGET.value], axis=1)
        return y  # type: ignore

    def __generate_x(self) -> np.ndarray:
        """Create x ie. numpy array the features, without target.

        Returns:
            np.ndarray: numpy ndarray of the features without target.
        """
        # Combine all the features into a numpy array
        xs = [self.df[f].values.tolist() for f in self.df.columns]  # type: ignore
        x = np.concatenate(xs, axis=1)

        # replace all nan values with 0
        x = np.nan_to_num(x, nan=0)

        return x

    def __encode_array_features(self) -> None:
        """Encode the array features. They must be numpy arrays."""
        # flatten 2d arrays to 1d
        for feature in Feature.array_2d_features() - self.discard_features:
            self.df[feature] = self.df[feature].apply(lambda row: sum(row, []))

        for feature in Feature.array_features() - self.discard_features:
            # remove nones that might occur in the arrays
            self.df[feature] = self.df[feature].apply(
                lambda arr: [elem for elem in arr if elem is not None]
            )
            # pad the arrays, so they are all the same length, necessary for sklearn
            self.df[feature] = pad_sequences(self.df[feature], padding="post").tolist()
            # Make each array a numpy array
            self.df[feature] = self.df[feature].apply(lambda arr: np.array(arr))

    def __encode_single_value_features(self) -> None:
        """Encode the non array features. They must be numpy arrays."""
        to_encode = Feature.array_features().not_in(self.df.columns)
        for f in to_encode:
            self.df[f] = self.df[f].apply(lambda val: np.array([val]))

    def __encode_categorical_features(self) -> None:
        """One-hot encode categorical features."""
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

        # Calculate the index for splitting the dataset
        split_idx = int(len(x) * self.test_size)

        # Split the dataset into training and testing sets
        x_test, y_test = x[:split_idx], y[:split_idx]
        x_train, y_train = x[split_idx:], y[split_idx:]

        return x_train, x_test, y_train, y_test
