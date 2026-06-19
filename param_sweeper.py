import os
import pandas as pd
import numpy as np
import torch
from models.oracle.attention_net import TemporalAttentionOracle
from sklearn.preprocessing import StandardScaler

def precompute_probabilities(df: pd.DataFrame, oracle_path: str) -> pd.DataFrame:
    """Passes the raw data through the frozen Oracle to get the AI probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exclude_cols = ['target', 'time', 'datetime', 'date']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('env_')]
    
    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    oracle.eval()

    scaler = StandardScaler()
    raw_features = scaler.fit_transform(df[feature_cols].values)
    probs_list = np.zeros((len(df), 3))
    
    with torch.no_grad():
        for i in range(30, len(df)):
            window = raw_features[i-30:i]
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            logits = oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_list[i] = probs
            
    df['prob_hold'] = probs_list[:, 0]
    df['prob_long'] = probs_list[:, 1]
    df['prob_short'] = probs_list[:, 2]
    
    return df

def run_sweep():
    print("=== XAU RL Engine V2: Parameter Sweeper ===")
    features_path = "data/processed/labeled_features_15m.csv"
    oracle_path = "models/oracle/best_oracle.pth"
    
    if not os.path.exists(oracle_path):
        print(f"ERROR: Oracle weights not found at {oracle_path}")
        return

    print("1. Loading Data & Computing Oracle Probabilities...")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Use the test split (last 20%) to see how it behaves out-of-sample
    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:].copy()
    test_df = precompute_probabilities(test_df, oracle_path)
    
    total_days = len(test_df) / 96.0  # 96 15-minute candles in a 24-hour period
    cooldown_duration = 24
    
    thresholds_to_test = [0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55]
    
    print(f"\nSimulation Period: ~{total_days:.1f} Days of Market Data")
    print("-" * 65)
    print(f"{'Threshold':<15} | {'Total Trades':<15} | {'Trades / Day':<15}")
    print("-" * 65)
    
    for threshold in thresholds_to_test:
        cooldown_timer = 0
        total_trades = 0
        
        for i in range(30, len(test_df)):
            if cooldown_timer > 0:
                cooldown_timer -= 1
                continue
                
            prob_long = test_df['prob_long'].iloc[i]
            prob_short = test_df['prob_short'].iloc[i]
            
            if prob_long >= threshold or prob_short >= threshold:
                total_trades += 1
                cooldown_timer = cooldown_duration
                
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        print(f"Gate: {threshold:<9} | Count: {total_trades:<10} | Freq: {trades_per_day:.2f}/day")
    
    print("-" * 65)
    print("Recommendation: Select the threshold that yields ~1.00 to 2.00 Trades / Day.")

if __name__ == "__main__":
    run_sweep()