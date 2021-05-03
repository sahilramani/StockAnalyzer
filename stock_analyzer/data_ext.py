# Helper functions to do some common operations on dataframes and arrays
import math
import pandas as pd


# Drops specified columns from the given dataframe
def  drop_columns(df:pd.DataFrame, *columns:list) -> pd.DataFrame :
    # Now we just drop the other fields in the dataset because we technically don't need them.
    return df.drop(columns=list(columns))


# Drops the columns except the ones from the given dataframe
def drop_columns_except(df:pd.DataFrame, *columns:list) -> pd.DataFrame :
    # Now we just drop the other fields in the dataset because we technically don't need them.
    return df.drop(columns=list(set(df.columns)-set(columns)))

# Given a set of counts, this method will split it according to the given percentages
def split_counts(data_len:int, seq_length:int, train_per:float, val_per:float) -> (int, int, int):
    remain_len = data_len
    train_count = math.floor(remain_len * train_per / seq_length) * seq_length
    remain_len -= train_count
    val_count = math.floor(remain_len * (val_per / (1.0 - train_per)) / seq_length) * seq_length
    remain_len -= val_count
    test_count = math.floor(remain_len / seq_length) * seq_length
    return (train_count, val_count, test_count)