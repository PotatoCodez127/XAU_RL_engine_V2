import pandas as pd
import numpy as np
from typing import List, Dict

class WalkForwardPipeline:
    def __init__(self, xau_path: str, dxy_path: str, embargo_bars: int = 200):
        """
        embargo_bars: Number of periods to skip between train and test 
        to prevent look-ahead bias from overlapping 4H/30m structural zones.
        """
        self.xau_path = xau_path
        self.dxy_path = dxy_path
        self.embargo_bars = embargo_bars
        self.master_df = None

    def load_and_merge(self) -> pd.DataFrame:
        # Load datasets, assuming 'time' is the primary column
        xau = pd.read_csv(self.xau_path, parse_dates=['time'], index_col='time')
        dxy = pd.read_csv(self.dxy_path, parse_dates=['time'], index_col='time')

        # Left join to maintain XAU as the master timeline
        df = xau.join(dxy, rsuffix='_dxy', how='left')

        # Forward fill missing DXY values to preserve rows before dropping 
        # the unfillable initial NaNs.
        df.ffill(inplace=True)
        df.dropna(inplace=True) 

        self.master_df = df
        return df

    def generate_splits(self, train_size: int, test_size: int, step_size: int) -> List[Dict[str, pd.DataFrame]]:
        """
        Generates rolling windows for Walk-Forward Analysis.
        """
        if self.master_df is None:
            raise ValueError("Data not loaded. Call load_and_merge() first.")

        splits = []
        total_len = len(self.master_df)

        for i in range(0, total_len - train_size - test_size, step_size):
            train_end = i + train_size
            test_start = train_end + self.embargo_bars
            test_end = test_start + test_size

            # Ensure we don't exceed the dataset bounds
            if test_end > total_len:
                break

            train_df = self.master_df.iloc[i:train_end]
            test_df = self.master_df.iloc[test_start:test_end]

            splits.append({'train': train_df, 'test': test_df})

        return splits