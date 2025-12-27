import pandas as pd
import numpy as np
from typing import Literal
import os

def load_data() -> pd.DataFrame:
    """
    Load and construct the main analysis dataset by merging:
    - Federal Funds Rate data
    - FOMC decision information
    - USD/VND forward rates
    - USD/VND spot rates

    The function performs:
    - Date alignment across all data sources
    - Cleaning of missing values
    - Log and log-difference transformations for exchange rates

    Returns:
        pd.DataFrame: Final cleaned and merged dataset ready for
        econometric analysis (event study / time series regression).
    """
    df = load_fed_funds_rate_data()
    df = load_fomc_data(df)
    df = process_fw_spot_data(df, "..\\data\\FW.xlsx", "FW", 17)
    df = process_fw_spot_data(df, "..\\data\\Spot.xlsx", "Spot", 27)
    return df


def load_fed_funds_rate_data() -> pd.DataFrame :
    """
    Load and preprocess Federal Funds Rate data.

    Operations:
    - Read data from Excel
    - Rename columns to standardized names
    - Convert date column to datetime format

    Returns:
        pd.DataFrame: DataFrame containing daily Federal Funds Rate
        with columns ['date', 'fed_funds'].
    """
    df = pd.read_excel('../data/fed_funds.xlsx')
    rename = {
        'observation_date': 'date',
        'DFF': 'fed_funds'
    }
    df = df.rename(columns=rename)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_fomc_data(df: pd.DataFrame) -> pd.DataFrame :
    """
    Merge FOMC meeting outcomes into the main DataFrame.

    Operations:
    - Load FOMC decision data from Excel
    - Encode qualitative FOMC decisions into numerical dummy variables:
        decrease → 1
        maintain → 2
        increase → 3
    - Merge FOMC information by date

    Args:
        df (pd.DataFrame): DataFrame containing date index.

    Returns:
        pd.DataFrame: Updated DataFrame including FOMC decision variables.
    """
    fomc_df = pd.read_excel('../data/fomc.xlsx')

    def get_fomc_action_dummy(fomc_change: str) -> int :
        if pd.isna(fomc_change):
            return np.nan
        mapping = {
            'decrease': 1,
            'maintain': 2,
            'increase': 3
        }
        return mapping.get(fomc_change, np.nan)
    
    fomc_df['fomc_action_dummy'] = fomc_df['fomc_change'].apply(lambda x: get_fomc_action_dummy(x))
    df = pd.merge(df, fomc_df, on='date', how='left')
    return df

def process_fw_spot_data(
    df: pd.DataFrame,
    filename: str,
    rate_type: Literal["FW", "Spot"], 
    skip_rows: int
) -> pd.DataFrame:
    """
    Process forward and spot rate data from Excel files.

    Args:
        df (pd.DataFrame): Original DataFrame to merge data into.
        filename (str): Path to the Excel file containing forward or spot rates.
        rate_type (str): Type of rate ('FW' or 'Spot').
        skip_rows (int): Number of rows to skip when reading the Excel file.

    Returns:
        pd.DataFrame: Updated DataFrame with merged forward or spot rate data.
    """

    sheets = pd.ExcelFile(filename).sheet_names
    for sheet in sheets: 

        data_df = pd.read_excel(filename, sheet_name=sheet, skiprows=skip_rows)
        name_mapping = {
            "Exchange Date": "date",
            "Bid": f"Bid_{rate_type}_{sheet.replace(' ', '_')}",
            "Ask": f"Ask_{rate_type}_{sheet.replace(' ', '_')}",
        }

        data_df.rename(columns=name_mapping, inplace=True)

        df = pd.merge(df, data_df[list(name_mapping.values())], on="date", how="left")
        if rate_type == 'FW':
            required_cols = [f'Bid_{rate_type}_usd_vnd_2w']
        else:
            required_cols = [f'Bid_{rate_type}_usd_vnd']
        df = df.dropna(subset=required_cols)

        for col_type in ["Bid", "Ask"]:
            col_name = f"{col_type}_{rate_type}_{sheet.replace(' ', '_')}"
            if col_name in df.columns:
                df[f"log_{col_name}"] = np.log(df[col_name].replace(0, np.nan))
                df[f"delta_log_{col_name}"] = df[f"log_{col_name}"].diff()
    return df 