import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from statistics import mean
from scipy.stats import norm

def flatten(l):
    return [item for sublist in l for item in sublist]

def flatten_and_floor():


def plot_speed_graf_for_segment(df: pd.DataFrame, index_list: list[int]) -> None:
    """
    CONDITION: the df must have osm_id as index. Creates a file of the distribution of speeds (floored) on a given segment.
    Args:
        df: a dataframe with osm_id as index
        index: the index you want to plot

    """
    #df = df.loc[index]



    for index in index_list:
        speeds_flat = flatten(df["speeds"].loc[index])

    #speeds_flat = flatten(df["speeds"].iloc[0])
    floor_values = []
    for list in speeds_flat:
        floor_values += [math.floor(x) for x in list]

    #floor_values = [math.floor(x) for x in l in speeds_flat]

    # Get dictionary with {key: no. of occurences}
    value_dic = Counter(floor_values)
    # Convert to dictionary to list
    list_dic = list(value_dic.items())
    # Sort accending wrt. key
    list_dic.sort()

    x = []
    y = []

    for (speed, occurences) in list_dic:
        x.append(speed)
        y.append(occurences)

    plt.plot(x, y)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Number of occurences on segment')
    plt.title(f"Speeds on segment {index_list[0]}-{index_list[-1]}")
    plt.savefig(f"./speed_figures/speed_graph_osm_id_{index_list[0]}-{index_list[-1]}.png")

def main():
    # Load in data:
    path_root = "/share-files/pickle_files_features_and_ground_truth"
    df: pd.DataFrame = (
        pd.read_pickle(path_root + "/2014.pkl")
    )

    # df = pd.read_pickle("one_row.pkl")

    # Set index
    df = df.set_index('osm_id')

    # Remove duplicates
    # (from: https://jianan-lin.medium.com/typeerror-unhashable-type-list-how-to-drop-duplicates-with-lists-in-pandas-4)
    no_duplicates = df[df.columns].loc[df[df.columns].astype(str).drop_duplicates().index]



    universitet_b_80 = [8149020, 8149021, 10240935, 10240932, 682803169, 682803170, 682803171, 682803172, 287131331, 10240934]

    plot_speed_graf_for_segment(no_duplicates, universitet_b_80)

if __name__ == "__main__":
    main()

def spand():
    return 0
    # Get rows with id
    #df = df.loc[df['osm_id'] == 8149020]

    # Remove duplicates
    # (from: https://jianan-lin.medium.com/typeerror-unhashable-type-list-how-to-drop-duplicates-with-lists-in-pandas-4)
    #no_duplicates = df[df.columns].loc[df[df.columns].astype(str).drop_duplicates().index]
    #no_duplicates.to_pickle("one_row.pkl")
    #no_duplicates = pd.read_pickle("one_row.pkl")

    # Plot normal distribution

    #speeds_flat = sum(no_duplicates["speeds"], [])
    # speeds_flat = flatten(no_duplicates["speeds"].iloc[0])
    #
    # # float("{:.2f}".format(x))
    # floor_values = [math.floor(x) for x in speeds_flat]
    # value_dic = Counter(floor_values)
    #
    # x = []
    # y = []
    #
    # list_dic = list(value_dic.items())
    # list_dic.sort()
    # print(list_dic)
    #
    # for (speed, occurences) in list_dic:
    #     x.append(speed)
    #     y.append(occurences)
    #
    # plt.plot(x, y)
    # plt.show()
    # plt.savefig("test.png")


    # Make sure to close the plt object once done
    #plt.close()


    #uni_b = df.loc[df['osm_id'] == 8149020]
    #index = []


    #duplicate_mask = uni_b.du

    #print(duplicate_mask)

    # print(uni_b["hast_gaeldende_hast"])

    # uni_b.to_csv("uni_b.csv")
    #print(uni_b)


    # # df = df.set_index('osm_id')
    # # universitetsboulevarden = df.iloc[8149020]
    #
    # speeds = universitetsboulevarden["speeds"]
    #
    # print(speeds)

    # for col in df.columns:
    #     print(col)



    # # Filter out any the coordinates of any trip, that are not on selected OSM_ID:
    # filtered_df = filter_segments(
    #     df, 8149020
    # )  # <- first segment on Universitetsboulevarden, 80km/t
    #
    # # Remove all unrelated trips:
    # filtered_trips_df = filter_trips(filtered_df)
    #
    # # Create columns containing average speeds through the segment
    # calculate_distance(filtered_trips_df)
    # filtered_trips_df["speeds"] = filtered_trips_df["distances"].apply(ms_to_kmh)
    # filtered_trips_df["avg_speed"] = filtered_trips_df["speeds"].apply(mean)  # type: ignore
    #
    # # Plot all trips, through selected segment, with their average speed in a histogram.
    # average_speeds = pd.Series(filtered_trips_df["avg_speed"])
    # plot_normal_distribution(average_speeds)