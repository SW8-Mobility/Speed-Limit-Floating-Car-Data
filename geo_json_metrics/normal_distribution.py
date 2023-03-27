import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gps_metrics import filter_segments, filter_trips, calculate_distance, ms_to_kmh
from statistics import mean
from scipy.stats import norm


def plot_normal_distribution(series: pd.Series) -> None:
    """Fits the average speeds into a normal distribution,
    using the method described here:
    https://www.geeksforgeeks.org/how-to-plot-normal-distribution-over-histogram-in-python/
    """
    data = np.random.normal(series)
    mu, std = norm.fit(data)
    plt.hist(data, bins=25, density=True, alpha=0.6, color="b")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)
    title = "Fit Values: {:.2F} and {:.2F}. Observations: {:d}".format(
        mu, std, len(series)
    )
    plt.title(title)
    plt.show()


def main():
    # Load in data:
    path_root = "./data/pickle_files"
    df: pd.DataFrame = (
        pd.read_pickle(path_root + "/2014.pkl").infer_objects().head(1000)
    )
    df = (
        df.reset_index()
    )  # The data frames indexes are, before reset, 0 to 9 repeatedly
    # Filter out any the coordinates of any trip, that are not on selected OSM_ID:
    filtered_df = filter_segments(
        df, 8149020
    )  # <- first segment on Universitetsboulevarden, 80km/t
    # Remove all unrelated trips:
    filtered_trips_df = filter_trips(filtered_df)
    # Create columns containing average speeds through the segment
    calculate_distance(filtered_trips_df)
    filtered_trips_df["speeds"] = filtered_trips_df["distances"].apply(ms_to_kmh)
    filtered_trips_df["avg_speed"] = filtered_trips_df["speeds"].apply(mean)  # type: ignore
    # Plot all trips, through selected segment, with their average speed in a histogram.
    average_speeds = pd.Series(filtered_trips_df["avg_speed"])
    plot_normal_distribution(average_speeds)


if __name__ == "__main__":
    main()
