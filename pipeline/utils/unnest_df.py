import pandas as pd # type: ignore

def unnest_df(df: pd.DataFrame, nest_header: str, key_list: list[str]) -> None:
    """
    Unnest a given key from a column, if the format is like a dictionary. The column is added to the dataframe and the
    name will be in lower case.
    Args:
        df: a dataframe
        nest_header: the column where the nested values are
        key_list: a list of keys to look for
    """
    for key in key_list:
        df[key.lower()] = df[nest_header].apply(lambda dict: dict[key])
