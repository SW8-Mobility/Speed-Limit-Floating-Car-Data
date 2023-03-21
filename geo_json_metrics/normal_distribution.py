import pandas as pd
from gps_metrics import filter_segments, filter_trips, calculate_distance, calculate_metrics

def main():
    # Load csv
    df: pd.DataFrame = pd.read_csv("./data/csv_files/2013.csv").infer_objects()
    print("\noriginal coordinates:")
    print(df.loc[:,"coordinates"])
    
    print("\noriginal df")
    print(df)
    filtered_df = filter_segments(df, 218939135)
    
    print("\nfiltered segments:")
    print(filtered_df.loc[:,"coordinates"])
    filtered_df = filter_trips(filtered_df)
    
    print("\nfiltered trips:")
    print(filtered_df)
    # calculate_distance(filtered_df)
    
    # my_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2])
    # my_data.hist()

if __name__ == "__main__":
    main()
