from typing import Any

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import math
from collections import Counter


def select_osm_rows(df: pd.DataFrame, index_list: list[int]) -> pd.DataFrame:
    """
    Select rows with osm_id in index list.
    Args:
        df: a dataframe with osm_id
        index_list: a list of osm_id

    Returns: dataframe containing rows with given osm_id

    """
    selected_rows = df.loc[df["osm_id"].isin(index_list)]
    return selected_rows


def flatten(nested_list: list[list[any]]) -> list[Any]:
    """
    Flatten a list such that [[1], [2], [3, 4]] becomes [1, 2, 3, 4]
    Args:
        nested_list: a list to flatten

    Returns: flattend list

    """
    return [item for sublist in nested_list for item in sublist]


def flatten_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten speeds column in dataframe using flatten()
    Args:
        df: dataframe with 'speeds'

    Returns: dataframe where speeds column now contains flattend lists

    """
    df["speeds"] = df["speeds"].apply(lambda l: flatten(l))
    return df


def flatten_and_concat_speeds(df: pd.DataFrame) -> list[float]:
    """
    Concatinate all speeds of a given dataframe and return as list.
    ASSUMPTION: speeds are flat lists
    Args:
        df: a dataframe with 'speeds' column

    Returns: list of all speeds in dataframe

    """
    df["speeds"] = df["speeds"].apply(lambda l: flatten(l))

    speed_list = []
    df["speeds"].apply(lambda l: speed_list.extend(l))

    return speed_list


def floor_list(input_list: list[float]) -> list[int]:
    """
    Floor all elements in list
    Args:
        input_list: a list of floats

    Returns: list of ints

    """
    return list(map(lambda x: math.floor(float(x)), input_list))


def plot_speed_graf_for_segment(
    speed_list: list[int], index_list: list[int], custom_title: str | None
) -> None:
    """
    Plot speed graph for a list of segments. Adds file to speed_figures folder
    Args:
        speed_list: a list of all speeds for osm_id's in index_list
        index_list: a list of osm_id's. Only used in order to name graph and files
        custom_title: a title of graph. If None, the index_list is used for naming

    """
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

    if custom_title is None:
        custom_title = f"Speeds on segment {index_list[0]}-{index_list[-1]}"

    plt.title(custom_title)
    plt.show()
    plt.savefig(
        f"./speed_figures/speed_graph_osm_id_{index_list[0]}-{index_list[-1]}.png"
    )
    plt.savefig(
        f"/speed_figures/speed_graph_osm_id_{index_list[0]}-{index_list[-1]}.png"
    )


def create_speed_graph(
    df: pd.DataFrame, osm_id_list: list[int], custom_title: str | None = None
) -> None:
    """
    Create a speed graph from a dataframe and list of osm_id's
    Args:
        df: a dataframe containing the rows to plot
        osm_id_list: a list of osm_ids to plot
        custom_title: a title of the graph. Default is None

    """

    # pick relevant rows
    df = select_osm_rows(df, osm_id_list)

    # Remove duplicates
    # (from: https://jianan-lin.medium.com/typeerror-unhashable-type-list-how-to-drop-duplicates-with-lists-in-pandas-4)
    df = df[df.columns].loc[df[df.columns].astype(str).drop_duplicates().index]

    # flatten and concat speed values
    speed_list = flatten_and_concat_speeds(df)

    # floor values
    speed_list_floored = floor_list(speed_list)

    # create graph
    plot_speed_graf_for_segment(speed_list_floored, osm_id_list, custom_title)


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

    create_speed_graph(df, universitet_b_80, "Universitetsboulevarden (80 km/h)")


if (
    __name__ == "__main__"
):  # Run it from the terminal!! Otherwise, the graph will be blank
    main()
