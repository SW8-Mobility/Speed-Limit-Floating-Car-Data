from pipeline.preprocessing.compute_features.feature import Feature
from sklearn.model_selection import train_test_split
import pandas as pd

pd.options.display.width = 0


def get_fake_input():
    columns = ["target"]
    for feature in Feature:
        if (
            feature != Feature.ROAD_TYPE
            and feature != Feature.ID
            and feature != Feature.COORDINATES
            and feature != Feature.LEVEL
            and feature != Feature.DAY_OF_WEEK
            and feature != Feature.TIME_GROUP
            and feature != Feature.VCR
        ):
            columns.append(feature.value)

    input_df = pd.DataFrame(
        data=[
            [
                50,
                [[10, 30, 40], [30, 40, 50], [50, 43, 46]],
                [[150, 2000, 100], [3000, 400, 150], [5000, 430, 4600]],
                [26.67, 40.0, 46.33],
                37.67,
                [10, 30, 43],
                10,
                [40, 50, 43],
                50,
                [[20, 35], [35, 45], [46.5, 44.5]],
                [30, 40, 46],
                40,
            ]
        ],
        columns=columns,
    )

    return input_df


def run_models():
    input_df = get_fake_input()

    data = [[123, "testvej", "testvej_osm", 123]]
    columns = ["osm_id", "cpr_name", "osm_name", "id"]
    output_df = pd.DataFrame(data=data, columns=columns)

    X = input_df.drop(columns=["target"])
    y = input_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def main():
    run_models()


if __name__ == "__main__":
    main()
