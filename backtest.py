import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.attention_net import TemporalAttentionOracle
from sklearn.preprocessing import StandardScaler

def precompute_probabilities(df: pd.DataFrame, oracle_path: str) -> pd.DataFrame:
    """Passes the raw data through the frozen Oracle to get the AI probabilities."""
    print("Loading Oracle...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exclude_cols = ['target', 'time', 'datetime', 'date']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('env_')]
    
    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    oracle.eval()

    print("Calculating AI Probabilities...")
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

def generate_report(journal_df: pd.DataFrame, equity_curve: list):
    """Calculates and prints professional trading statistics."""
    os.makedirs("logs", exist_ok=True)
    
    if len(journal_df) == 0:
        print("\nNo trades were taken during the backtest period.")
        return

    # Separate Wins and Losses
    wins = journal_df[journal_df['PnL_$'] > 0]
    losses = journal_df[journal_df['PnL_$'] <= 0]
    
    total_trades = len(journal_df)
    winrate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

    # Calculate $ Metrics
    avg_win_usd = wins['PnL_$'].mean() if len(wins) > 0 else 0
    max_win_usd = wins['PnL_$'].max() if len(wins) > 0 else 0
    min_win_usd = wins['PnL_$'].min() if len(wins) > 0 else 0

    avg_loss_usd = losses['PnL_$'].mean() if len(losses) > 0 else 0
    max_loss_usd = losses['PnL_$'].min() if len(losses) > 0 else 0 # Minimum is the biggest negative number
    min_loss_usd = losses['PnL_$'].max() if len(losses) > 0 else 0 # Maximum negative is the smallest loss
    
    # Calculate Multiplier Metrics (Proxy for Pips/Distance)
    avg_tp_mult = wins['Take_Profit_Mult'].mean() if len(wins) > 0 else 0
    max_tp_mult = wins['Take_Profit_Mult'].max() if len(wins) > 0 else 0
    min_tp_mult = wins['Take_Profit_Mult'].min() if len(wins) > 0 else 0
    
    avg_sl_mult = losses['Stop_Loss_Mult'].mean() if len(losses) > 0 else 0
    max_sl_mult = losses['Stop_Loss_Mult'].max() if len(losses) > 0 else 0
    min_sl_mult = losses['Stop_Loss_Mult'].min() if len(losses) > 0 else 0

    print("\n" + "="*40)
    print(" 🚀 XAU RL V2 PERFORMANCE REPORT 🚀")
    print("="*40)
    print(f"Total Trades Taken:   {total_trades}")
    print(f"Winrate:              {winrate:.2f}%")
    print(f"Final Equity:         ${equity_curve[-1]:.2f}")
    print(f"Max Drawdown:         {journal_df['Drawdown_%'].max():.2f}%\n")
    
    print("--- TAKE PROFIT METRICS ---")
    print(f"Average Win:          +${avg_win_usd:.2f} (Avg TP Mult: {avg_tp_mult:.2f}x)")
    print(f"Largest Win:          +${max_win_usd:.2f} (Max TP Mult: {max_tp_mult:.2f}x)")
    print(f"Smallest Win:         +${min_win_usd:.2f} (Min TP Mult: {min_tp_mult:.2f}x)\n")
    
    print("--- STOP LOSS METRICS ---")
    print(f"Average Loss:         ${avg_loss_usd:.2f} (Avg SL Mult: {avg_sl_mult:.2f}x)")
    print(f"Largest Loss:         ${max_loss_usd:.2f} (Max SL Mult: {max_sl_mult:.2f}x)")
    print(f"Smallest Loss:        ${min_loss_usd:.2f} (Min SL Mult: {min_sl_mult:.2f}x)")
    print("="*40)

    # Generate Equity Curve Chart
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve", color='#00ffcc', linewidth=2)
    plt.fill_between(range(len(equity_curve)), equity_curve, min(equity_curve) * 0.99, color='#00ffcc', alpha=0.1)
    
    plt.title("RL Agent Equity Curve (Out of Sample)", fontsize=16, color='white')
    plt.xlabel("Steps (15m Candles)", fontsize=12, color='white')
    plt.ylabel("Account Balance ($)", fontsize=12, color='white')
    plt.grid(color='#333333', linestyle='--', linewidth=0.5)
    
    # Dark Mode Styling
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        
    plt.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
    
    chart_path = "logs/equity_curve.png"
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"\n📈 Equity Curve chart saved to: {chart_path}")
    
    journal_path = "logs/final_backtest_journal.csv"
    journal_df.to_csv(journal_path, index=False)
    print(f"📓 Detailed trade journal saved to: {journal_path}")

def run_backtest():
    print("=== XAU RL V2 Backtest Engine ===")
    
    features_path = "data/processed/labeled_features_15m.csv"
    oracle_path = "models/oracle/best_oracle.pth"
    manager_path = "models/manager/saved/wfa_55/best_model.zip" 
    
    if not os.path.exists(manager_path):
        print(f"ERROR: Could not find final manager weights at {manager_path}")
        return

    raw_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Backtest on the last 20% of data (Out of Sample)
    test_size = int(len(raw_df) * 0.2)
    test_df = raw_df.iloc[-test_size:].copy()
    
    enriched_df = precompute_probabilities(test_df, oracle_path)
    
    print("Initializing SAC Manager...")
    env = XAUDynamicEnv(df=enriched_df)
    model = SAC.load(manager_path, env=env)
    
    print("Running Simulation...")
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    journal = []
    equity_curve = [10000.0] # Starting balance
    
    while not (terminated or truncated):
        previous_balance = env.balance
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_balance = env.balance
        equity_curve.append(current_balance)
        
        direction_val = 0
        if action[0] > 0.33: direction_val = 1
        elif action[0] < -0.33: direction_val = 2
        
        if direction_val != 0:
            trade_pnl = current_balance - previous_balance
            
            journal.append({
                "Step": env.current_step,
                "Action": "Long" if direction_val == 1 else "Short",
                "Position_Size_%": round(((action[1] + 1) / 2) * 5, 2),
                "Take_Profit_Mult": round(info.get('tp_mult_used', 0), 2),
                "Stop_Loss_Mult": round(info.get('sl_mult_used', 0), 2),
                "PnL_$": round(trade_pnl, 2),
                "Equity": round(current_balance, 2),
                "Drawdown_%": round(info.get('drawdown', 0) * 100, 2)
            })

    # Generate the beautiful output
    generate_report(pd.DataFrame(journal), equity_curve)

if __name__ == "__main__":
    run_backtest()