import numpy as np
from sklearn.compose import make_column_transformer

from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd  # type: ignore
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

class SKFormatter:

    def __init__(self, dataset_path: str, target_feature: Feature):
        self.df = pd.read_pickle(dataset_path)
        self.target = target_feature.value

        # features we don't want to train with
        self.discard_features = [Feature.OSM_ID.value]

    def generate_train_test_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.df = self.__encode_categorical_features()
        self.__encode_single_value_features(self.df)

        self.df = self.df.rename(columns={Feature.HAST_GAELDENDE_HAST.value: Feature.TARGET.value})
        y = self.df[Feature.TARGET.value].values
        self.df = self.df.drop([Feature.TARGET.value], axis=1)

        x = self.__generate_x()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        return x_train, x_test, y_train, y_test


    def __generate_x(self) -> tuple[np.ndarray, np.ndarray]:
        # don't train with the following features
        self.df = self.df.drop(self.discard_features, axis=1)

        # make 2d array features 1d by flattening them
        for feature in Feature.array_2d_features():
            self.df[feature] = self.df[feature].apply(lambda row: sum(row, []))

        for feature in Feature.array_1d_features() + Feature.array_2d_features():
            # remove nones that might occur in the arrays
            self.df[feature] = self.df[feature].apply(lambda arr: [elem for elem in arr if arr is not None])
            # pad the arrays, so they are all the same length, necessary for sklearn
            self.df[feature] = pad_sequences(self.df[feature], padding='post').tolist()
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
            (OneHotEncoder(), Feature.categorical_features()),
            remainder='passthrough')

        transformed = transformer.fit_transform(self.df)
        self.df = pd.DataFrame(transformed, columns=transformer.get_feature_names())

    def test_train_split(self):
        return