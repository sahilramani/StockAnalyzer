# Load a CSV file populated with stock data
import pandas as pd
import datetime
from dateutil import parser
from os import path

# Expects csv data with (at least) the following fields: 'datetime'
# Returns loaded csv data with all columns, always ascending
def load_file(file:str, alwaysAscending:bool=True) -> pd.DataFrame:
    if not path.isfile(file):
        raise ValueError(f"{file} is not a file")
    
    raw_data = pd.read_csv(file)
    
    if 'datetime' not in raw_data.columns:
        raise ValueError(f"{file} does not have a column for 'datetime'")
    if len(raw_data) < 2:
        raise ValueError(f"{file} contains less than two rows of data. We need at least 2 to proceed")
    
    if (alwaysAscending == True) and (parser.isoparse(raw_data.at[1, 'datetime']) - parser.isoparse(raw_data.at[0, 'datetime'])).total_seconds() < 0:
        # If we're ordered descending, let's flip that
        # Reverse the data frame since we get the data backwards
        raw_data = raw_data[::-1].reset_index(drop=True)

    if 'index' not in raw_data.columns:
        # Create a new index column
        raw_data.reset_index(level=0, inplace=True)
    
    return raw_data


# Loads a csv file given a symbol and other parameters
def load_symbol(root_dir:str, symbol:str, interval:str, year:str, alwaysAscending:bool=True) -> pd.DataFrame:
    filename = f'{symbol}_{year}_{interval}.csv'
    return load_file(path.join(root_dir, filename), alwaysAscending)
