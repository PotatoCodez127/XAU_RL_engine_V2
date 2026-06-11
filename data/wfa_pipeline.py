import pandas as pd
import numpy as np
from typing import List, Dict

class WalkForwardPipeline:
    def __init__(self, xau_path: str, dxy_path: str, embargo_bars: int = 200):
        """
        embargo_bars: Number of periods to skip between train and test 
        to prevent look-ahead bias from overlapping structural zones.
        """
        self.xau_path = xau_path
        self.dxy_path = dxy_path
        self.embargo_bars = embargo_bars
        self.master_df = None

    def _load_mt_csv(self, filepath: str) -> pd.DataFrame:
        """Parses MetaTrader specific CSV/TSV exports."""
        # MetaTrader files are often tab-delimited
        df = pd.read_csv(filepath, sep='\t')
        
        # If the file is actually comma-separated, fallback
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',')

        # Clean MetaTrader headers: <DATE> -> date
        df.columns = [c.strip('<>').lower() for c in df.columns]
        
        # Combine date and time into a single datetime index
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date', 'time'], inplace=True)
        
        return df

    def load_and_merge(self) -> pd.DataFrame:
        xau = self._load_mt_csv(self.xau_path)
        dxy = self._load_mt_csv(self.dxy_path)

        # Add prefix to DXY columns to avoid collision (e.g., close -> close_dxy)
        dxy.columns = [f"{c}_dxy" for c in dxy.columns]

        # Left join to maintain XAU as the master timeline
        df = xau.join(dxy, how='left')

        # Forward fill missing DXY values to preserve rows
        df.ffill(inplace=True)
        df.dropna(inplace=True) 

        self.master_df = df
        return df

    def generate_splits(self, train_size: int, test_size: int, step_size: int) -> List[Dict[str, pd.DataFrame]]:
        """Generates rolling windows for Walk-Forward Analysis."""
        if self.master_df is None:
            raise ValueError("Data not loaded. Call load_and_merge() first.")

        splits = []
        total_len = len(self.master_df)

        for i in range(0, total_len - train_size - test_size, step_size):
            train_end = i + train_size
            test_start = train_end + self.embargo_bars
            test_end = test_start + test_size

            if test_end > total_len:
                break

            train_df = self.master_df.iloc[i:train_end]
            test_df = self.master_df.iloc[test_start:test_end]

            splits.append({'train': train_df, 'test': test_df})

        return splits