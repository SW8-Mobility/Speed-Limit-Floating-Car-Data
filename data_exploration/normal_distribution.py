import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter


def select_speeds(df, index_list) -> pd.DataFrame:
    selected_rows = df.loc[df["osm_id"].isin(index_list)]
    return selected_rows


def flatten(l):
    return [item for sublist in l for item in sublist]


def flatten_speeds(df) -> pd.DataFrame:
    df["speeds"] = df["speeds"].apply(lambda l: flatten(l))
    return df


def concat_speeds(df):
    speed_list = []
    df["speeds"].apply(lambda l: speed_list.extend(l))

    return speed_list


def floor_list(input_list):
    return list(map(lambda x: math.floor(float(x)), input_list))


def plot_speed_graf_for_segment(speed_list: list[int], index_list: list[int]) -> None:
    value_dic = Counter(speed_list)
    # Convert to dictionary to list
    list_tuple = list(value_dic.items())
    # Sort accending wrt. key
    list_tuple.sort()

    x = []
    y = []

    for speed, occurences in list_tuple:
        x.append(speed)
        y.append(occurences)

    plt.plot(x, y)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Number of occurences on segment")
    plt.title(f"Speeds on segment {index_list[0]}-{index_list[-1]}")
    plt.show()
    plt.savefig(
        f"./speed_figures/speed_graph_osm_id_{index_list[0]}-{index_list[-1]}.png"
    )

def create_speed_graph(df, osm_id_list):
    # pick relevant rows
    df = select_speeds(df, osm_id_list)

    # Remove duplicates
    # (from: https://jianan-lin.medium.com/typeerror-unhashable-type-list-how-to-drop-duplicates-with-lists-in-pandas-4)
    df = df[df.columns].loc[df[df.columns].astype(str).drop_duplicates().index]

    # flatten speeds
    df = flatten_speeds(df)

    # concat values
    speed_list = concat_speeds(df)

    # floor values
    speed_list_floored = floor_list(speed_list)

    # create graph
    plot_speed_graf_for_segment(speed_list_floored, osm_id_list)

def main():
    # Load in data:
    path_root = "/share-files/pickle_files_features_and_ground_truth"
    df: pd.DataFrame = pd.read_pickle(path_root + "/2014.pkl")
    universitet_b_80 = [
        8149020,
        8149021,
        10240935,
        10240932,
        682803169,
        682803170,
        682803171,
        682803172,
        287131331,
        10240934,
    ]

    create_speed_graph(df, universitet_b_80)

if __name__ == "__main__":
    main()
