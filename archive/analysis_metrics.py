def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate aggregate min, max, and avg for dataframe

    Args:
        df (pd.DataFrame): Dataframe to calculate on

    Returns:
        tuple[float, float, float]: tuple containing the avg, min and max
    """
    # per entry: max, min, avg speed across entire segment, median
    df["avg_speed"] = df["speeds"].apply(mean)  # type: ignore
    df["max_speed"] = df["speeds"].apply(max)
    df["min_speed"] = df["speeds"].apply(min)
    df["median"] = df["speeds"].apply(median)  # type: ignore

    # for all entries: median_avg, median_min, median_max
    median_avg = df["avg_speed"].median()
    median_min = df["min_speed"].median()
    median_max = df["max_speed"].median()

    return (median_avg, median_min, median_max)
