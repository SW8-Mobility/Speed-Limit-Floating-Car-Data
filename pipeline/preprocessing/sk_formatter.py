from typing import Union
import numpy as np
from pipeline.preprocessing.compute_features.feature import Feature, FeatureList
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


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
        # Setting the index of the dataframe to be the osm_id
        

        self.full_dataset = full_dataset
        self.dataset_size = dataset_size
        self.test_size = test_size
        self.__remove_categorical_features()
        self.__prepare_df()


        self.discard_features: FeatureList = (
            FeatureList(  # default discard list, if no argument is provided
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

        self.target_feature: str = (  # default target_feature, if none provided
            Feature.HAST_GAELDENDE_HAST.value if target is None else target.value
        )

        self.params = self.__params()

    def __prepare_df(self):
        """Remove duplicates, set number of rows, and set index to osm_id."""
        self.__remove_duplicates()
        if not self.full_dataset:
            self.df = self.df.head(self.dataset_size)
        self.df["index"] = self.df[Feature.OSM_ID.value]
        self.df = self.df.set_index("index")

    def __remove_categorical_features(self) -> None:
        """Remove categorical features."""
        for feature in Feature.categorical_features():
            if feature in self.df.columns:
                self.df.drop([feature], inplace=True, axis=1)

    def __params(self) -> dict:
        """returns the parameters for sk_formatter,
        along with the columns for the dataframe before and
        after processing.
        This is useful, such that if parmeters are changed
        for features for our dataset is changed, we can find
        the original. A model trained on certain features, will
        not work, if features in the test set are not the same!

        Returns:
            dict: dictionary with parameters.
        """
        params = self.__dict__.copy()
        df = params.pop("df")
        params["input_df_columns"] = list(df.columns)
        return params

    def generate_train_test_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Generate the test train split.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
        """

        # don't train with the following features
        self.df = self.df.drop(self.discard_features, axis=1)

        # encode features
        # self.__encode_categorical_features()
        self.__encode_array_features()

        # save df config after processing
        self.params["processed_df_columns"] = list(self.df.columns)

        self.__replace_nones()

        # extract target
        y = self.__extract_y()

        # x_train, x_test, y_train, y_test
        return train_test_split(self.df, y, test_size=self.test_size, random_state=42)  # type: ignore

    def __replace_nones(self) -> None:
        """Replace all None values with 0"""
        self.df.fillna(0, inplace=True)

    def __remove_duplicates(self) -> None:
        """Remove duplicates from dataset.
        should ideally have been done before.
        """
        cols = self.df.columns
        self.df = self.df[cols].loc[self.df[cols].astype(str).drop_duplicates().index]
        ""

    def __extract_y(self) -> pd.Series:
        """Extract target from dataframe as y, and remove it
        from the df.

        Returns:
            pd.Series: Pandas series of target values.
        """
        self.df = self.df.rename(columns={self.target_feature: Feature.TARGET.value})
        y = self.df[Feature.TARGET.value]
        self.df = self.df.drop([Feature.TARGET.value], axis=1)
        return y  # type: ignore

    def __encode_array_features(self, col_num: int = 20) -> None:
        """Encode the array features. They must be numpy arrays.
        Arrays become seperate columns.

        Args:
            col_num (int, optional): How many columns are the arrays feature converted to. Defaults to 20.
        """
        # flatten 2d arrays to 1d
        for feature in Feature.array_2d_features() - self.discard_features:
            self.df[feature] = self.df[feature].apply(lambda row: sum(row, []))

        for feature in Feature.array_features() - self.discard_features:
            # remove nones that might occur in the arrays
            self.df[feature] = self.df[feature].apply(
                lambda arr: [elem for elem in arr if elem is not None]
            )
            cols = self.df
            s = cols.shape
            d = self.df[feature]
            # pad the arrays, so they are all the same length, necessary for sklearn
            self.df[feature] = pad_sequences(
                self.df[feature],
                padding="post",
                maxlen=col_num,
                truncating="post",
                dtype="float32",
            ).tolist()

            # split arrays into seperate columns
            new_cols = [f"{feature}_{i}" for i in range(col_num)]
            self.df[new_cols] = pd.DataFrame(
                self.df[feature].tolist(), index=self.df.index
            )

        # all features are encoded, drop the original
        self.df.drop(
            Feature.array_features() - self.discard_features, inplace=True, axis=1
        )

    def __encode_categorical_features(self) -> None:
        """DOES NOT WORK!"""
        # categorical_features = Feature.categorical_features() - self.discard_features
        # self.df = pd.get_dummies(self.df, columns=categorical_features, dtype=int)
        # self.df = pd.concat([one_hot_encoded, self.df], axis=1)
        # self.df = self.df.drop(categorical_features, axis=1)
        pass

    # def __encode_single_value_features(self) -> None:
    #     """Encode the non array features. They must be numpy arrays."""
    #     to_encode = Feature.array_features().not_in(self.df.columns)
    #     for f in to_encode:
    #         self.df[f] = self.df[f].apply(lambda val: np.array([val]))

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

    # def __test_train_split(
    #     self, x: np.ndarray, y: np.ndarray
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """Split x and y into a train and test sets.
    #     The dataset will always be split, such that the first
    #     test_size number of rows are used for testing, and the
    #     remaining are used for training. This is useful for stitching
    #     the predictions on the test set back to the original osm_ids.

    #     Args:
    #         x (np.ndarray): numpy array with features
    #         y (np.ndarray): numpy array with target

    #     Returns:
    #         tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, x_test, y_train, y_test
    #     """

    #     # Calculate the index for splitting the dataset
    #     split_idx = int(len(x) * self.test_size)

    #     # Split the dataset into training and testing sets
    #     x_test, y_test = x[:split_idx], y[:split_idx]
    #     x_train, y_train = x[split_idx:], y[split_idx:]

    #     return x_train, x_test, y_train, y_test

    # def __generate_x(self) -> np.ndarray:
    #     """Create x ie. numpy array the features, without target.

    #     Returns:
    #         np.ndarray: numpy ndarray of the features without target.
    #     """
    #     # Combine all the features into a numpy array
    #     xs = [self.df[f].values.tolist() for f in self.df.columns]  # type: ignore
    #     x = np.concatenate(xs, axis=1)

    #     # replace all nan values with 0
    #     x = np.nan_to_num(x, nan=0)

    #     return x
