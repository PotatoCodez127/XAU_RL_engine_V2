import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, xau_path: str, dxy_path: str):
        self.xau_path = xau_path
        self.dxy_path = dxy_path

    def _load_mt_csv(self, filepath: str) -> pd.DataFrame:
        """Parses MetaTrader specific CSV/TSV exports."""
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',')

        df.columns = [c.strip('<>').lower() for c in df.columns]
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
        df.set_index('datetime', inplace=True)
        
        # Drop MT5 specific fluff
        df.drop(columns=['date', 'time', 'tickvol', 'vol', 'spread'], inplace=True, errors='ignore')
        return df

    def resample_hloc(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        return df.resample(timeframe).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

    def calculate_daily_levels(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        daily = df_m1.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        daily['prev_high'] = daily['high'].shift(1)
        daily['prev_low'] = daily['low'].shift(1)
        daily['prev_close'] = daily['close'].shift(1)

        daily['daily_eq'] = (daily['prev_high'] + daily['prev_low']) / 2.0
        daily['pivot'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3.0
        daily['R1'] = (2 * daily['pivot']) - daily['prev_low']
        daily['S1'] = (2 * daily['pivot']) - daily['prev_high']

        daily_levels = daily[['daily_eq', 'pivot', 'R1', 'S1']].dropna()
        return pd.merge_asof(df_m1.sort_index(), daily_levels.sort_index(), left_index=True, right_index=True, direction='backward')

    def calculate_wick_zones(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = df.copy()
        df['rolling_max'] = df['high'].rolling(window=window*2 + 1, center=False).max()
        df['rolling_min'] = df['low'].rolling(window=window*2 + 1, center=False).min()
        
        is_swing_high = df['high'].shift(window) == df['rolling_max']
        is_swing_low = df['low'].shift(window) == df['rolling_min']
        
        for col in ['res_zone_top', 'res_zone_bottom', 'sup_zone_top', 'sup_zone_bottom']:
            df[col] = np.nan
            
        res_mask = is_swing_high
        df.loc[res_mask, 'res_zone_top'] = df['high'].shift(window)
        df.loc[res_mask, 'res_zone_bottom'] = df[['open', 'close']].shift(window).max(axis=1)
        
        sup_mask = is_swing_low
        df.loc[sup_mask, 'sup_zone_bottom'] = df['low'].shift(window)
        df.loc[sup_mask, 'sup_zone_top'] = df[['open', 'close']].shift(window).min(axis=1)
        
        for col in ['res_zone_top', 'res_zone_bottom', 'sup_zone_top', 'sup_zone_bottom']:
            df[col] = df[col].ffill()
            
        return df

    # --- NEW V3 MATH METHODS ---
    def _get_fractional_weights(self, d: float, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
        return np.array(w[::-1]).reshape(-1, 1)

    def apply_fractional_differentiation(self, series: pd.Series, d: float, window: int = 50) -> pd.Series:
        weights = self._get_fractional_weights(d, window)
        def frac_dot_product(x):
            if len(x) == window: return np.dot(x, weights)[0]
            return np.nan
        return series.rolling(window=window).apply(frac_dot_product, raw=True)
    # ---------------------------

    def convert_to_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        stationary_df = pd.DataFrame(index=df.index)
        
        # Keep raw prices for the RL Environment math
        stationary_df['env_open'] = df['open']
        stationary_df['env_high'] = df['high']
        stationary_df['env_low'] = df['low']
        stationary_df['env_close'] = df['close']
        
        # --- V3 UPGRADE: 14-Period ATR Calculation ---
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        stationary_df['env_atr'] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()
        
        # --- V3 UPGRADE: Fractional Differentiation (Replaces simple pct_change) ---
        stationary_df['close_frac_diff'] = self.apply_fractional_differentiation(df['close'], d=0.45, window=50)
        
        # --- V3 UPGRADE: ATR-Normalized Momentum (Replaces high_ret/low_ret) ---
        stationary_df['mom_1_norm'] = (df['close'] - df['close'].shift(1)) / stationary_df['env_atr']
        stationary_df['mom_4_norm'] = (df['close'] - df['close'].shift(4)) / stationary_df['env_atr']
        
        # Preserve DXY exactly as you had it
        if 'dxy_pct_change_15m' in df.columns:
            stationary_df['dxy_pct_change_15m'] = df['dxy_pct_change_15m']
            
        exclude_cols = ['open', 'high', 'low', 'close', 'dxy_pct_change_15m']
        price_level_cols = [c for c in df.columns if c not in exclude_cols]
        
        # --- V3 UPGRADE: ATR-Normalized Zone Distances ---
        # Instead of `(zone - close)/close`, we divide by ATR so the Oracle knows 
        # how far the zone is RELATIVE to current market volatility.
        for col in price_level_cols:
            stationary_df[f'dist_{col}_norm'] = (df[col] - df['close']) / stationary_df['env_atr']
            
        return stationary_df.dropna()

    def generate_labels(self, df: pd.DataFrame, max_hold: int = 32, rr_ratio: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        
        # Note: 'env_atr' is now pre-calculated in convert_to_stationary, so we just use it directly!
        close_p, high_p, low_p, atr_v = df['env_close'].values, df['env_high'].values, df['env_low'].values, df['env_atr'].values
        targets = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - max_hold):
            if np.isnan(atr_v[i]): continue
            
            entry_price, atr = close_p[i], max(atr_v[i], 0.5)
            spread = 0.15
            
            long_tp, long_sl = entry_price + (atr * rr_ratio) + spread, entry_price - atr - spread
            short_tp, short_sl = entry_price - (atr * rr_ratio) - spread, entry_price + atr + spread
            
            long_valid, short_valid, target = True, True, 0
            
            for j in range(1, max_hold + 1):
                f_high, f_low = high_p[i + j], low_p[i + j]
                
                if long_valid:
                    if f_low <= long_sl: long_valid = False 
                    elif f_high >= long_tp: target, long_valid = 1, False
                        
                if short_valid:
                    if f_high >= short_sl: short_valid = False 
                    elif f_low <= short_tp: target, short_valid = 2, False
                        
                if not long_valid and not short_valid: break 
                    
            targets[i] = target
            
        df['target'] = targets
        return df.dropna().iloc[:-max_hold].copy()

    def build_and_save(self, output_path: str):
        print("1. Loading raw M1 data...")
        df_m1 = self.calculate_daily_levels(self._load_mt_csv(self.xau_path).sort_index())
        
        print("2. Building 15m Base, Zones & EMA...")
        df_15m = self.resample_hloc(df_m1, '15min')
        df_15m[f'ema_50'] = df_15m['close'].ewm(span=50, adjust=False).mean()
        df_15m = self.calculate_wick_zones(df_15m, window=5).rename(columns=lambda x: f"{x}_15m" if 'zone' in x or 'rolling' in x else x)
        
        print("3. Mapping 30m and 4H structural zones...")
        df_30m = self.calculate_wick_zones(self.resample_hloc(df_m1, '30min'), window=5).rename(columns=lambda x: f"{x}_30m" if 'zone' in x else x)
        df_4h = self.calculate_wick_zones(self.resample_hloc(df_m1, '4h'), window=5).rename(columns=lambda x: f"{x}_4h" if 'zone' in x else x)
        
        print("4. Processing DXY correlation...")
        dxy_m1 = self._load_mt_csv(self.dxy_path).sort_index()
        dxy_15m = self.resample_hloc(dxy_m1, '15min')
        dxy_15m['dxy_pct_change_15m'] = dxy_15m['close'].pct_change()

        print("5. Stitching timelines together...")
        daily_cols = df_m1[['daily_eq', 'pivot', 'R1', 'S1']].resample('15min').last().ffill()
        master = pd.merge_asof(df_15m, daily_cols, left_index=True, right_index=True)
        master = pd.merge_asof(master, df_30m[[c for c in df_30m.columns if 'zone' in c]], left_index=True, right_index=True)
        master = pd.merge_asof(master, df_4h[[c for c in df_4h.columns if 'zone' in c]], left_index=True, right_index=True)
        master = pd.merge_asof(master, dxy_15m[['dxy_pct_change_15m']], left_index=True, right_index=True).dropna()
        
        print("6. Converting to Stationary Features (V3 Math)...")
        master = self.convert_to_stationary(master)
        
        print("7. Sweeping Future for Dynamic ATR Targets...")
        master_labeled = self.generate_labels(master)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        master_labeled.to_csv(output_path)
        print(f"SUCCESS: Labeled Master Dataset saved to {output_path}")

if __name__ == "__main__":
    XAU_PATH = "data/raw/XAUUSDr_M1_202306070100_202602202359.csv"
    DXY_PATH = "data/raw/USDIndex_M1_202001020300_202606112359.csv"
    OUTPUT_PATH = "data/processed/labeled_features_15m.csv"
    
    engineer = FeatureEngineer(XAU_PATH, DXY_PATH)
    engineer.build_and_save(OUTPUT_PATH)