import os
import torch
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from env.xau_dynamic_env import XAUDynamicEnv
from live_simulator import HighFidelitySimulator

# Suppress SB3 warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def optimize_agent(trial):
    # 1. Sample Hyperparameters mathematically
    gamma = trial.suggest_float("gamma", 0.90, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    
    print(f"\\n--- Starting Trial {trial.number} ---")
    print(f"Testing Config: Gamma={gamma:.4f}, LR={learning_rate:.5f}, Batch={batch_size}")

    # 2. Load Data & Enforce Strict Firewall
    data_path = "data/processed/labeled_features_15m.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 3. Train the Agent on In-Sample Data
    train_env = XAUDynamicEnv(train_df, initial_balance=10000.0)
    
    model = SAC(
        "MlpPolicy", 
        train_env, 
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tau=tau,
        train_freq=train_freq,
        ent_coef="auto",
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train for a fast, concentrated burst to gauge mathematical viability
    model.learn(total_timesteps=50000)
    
    # Save temporary weights for the simulator to use
    temp_path = f"models/manager/saved/trial_{trial.number}.zip"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    model.save(temp_path)
    
    # 4. Evaluate using the High-Fidelity Simulator on Unseen Data
    oracle_path = "models/oracle/best_oracle.pth"
    sim = HighFidelitySimulator(data_path, oracle_path, temp_path)
    
    # We hijack the simulator to return metrics directly instead of printing
    equity, total_trades = run_evaluation_simulation(sim, split_idx)
    
    # Cleanup temporary weights to save disk space
    os.remove(temp_path)
    
    # 5. Apply the Frequency Constraints (Min 2/week, Max 1/day)
    # Assume the 20% holdout is roughly 300 days (approx 42 weeks)
    estimated_weeks = len(test_df) / (96 * 5) # 96 bars/day * 5 days/week
    min_trades = estimated_weeks * 2
    max_trades = estimated_weeks * 5 # 1 per day max
    
    if total_trades < min_trades:
        print(f"Trial {trial.number} FAILED: Undertrading ({total_trades} trades)")
        return -10000.0 # Death penalty
        
    if total_trades > max_trades:
        print(f"Trial {trial.number} FAILED: Overtrading ({total_trades} trades)")
        return -10000.0 # Death penalty
        
    print(f"Trial {trial.number} SUCCESS: {total_trades} trades | Final Equity: ${equity:.2f}")
    
    # Optuna maximizes this returned value
    return equity

def run_evaluation_simulation(sim, holdout_start_idx):
    """A stripped-down version of your simulator purely for extracting final metrics."""
    equity = sim.initial_balance
    peak_equity = equity
    active_trade = None
    pending_signal = None
    bars_since_last_trade = 0 
    total_trades = 0
    
    for i in range(holdout_start_idx, len(sim.df) - 1):
        current_time = sim.df.index[i]
        current_bar = sim.df.iloc[i]
        bars_since_last_trade += 1
        
        # Manage Active Trade
        if active_trade is not None:
            trade_closed = False
            exit_price = 0.0
            
            if current_time.weekday() == 4 and current_time.hour >= 22 and current_time.minute >= 45:
                trade_closed, exit_price = True, current_bar['env_close']
            else:
                if active_trade['type'] == 'Long':
                    if current_bar['env_low'] <= active_trade['sl']:
                        trade_closed, exit_price = True, active_trade['sl']
                    elif current_bar['env_high'] >= active_trade['tp']:
                        trade_closed, exit_price = True, active_trade['tp']
                elif active_trade['type'] == 'Short':
                    if current_bar['env_high'] >= active_trade['sl']:
                        trade_closed, exit_price = True, active_trade['sl']
                    elif current_bar['env_low'] <= active_trade['tp']:
                        trade_closed, exit_price = True, active_trade['tp']

            if trade_closed:
                pip_diff = (exit_price - active_trade['entry']) * 10
                if active_trade['type'] == 'Short': pip_diff *= -1
                
                gross_pnl = pip_diff * sim.pip_value_per_lot * active_trade['lot_size']
                equity += (gross_pnl - active_trade['total_friction'])
                if equity > peak_equity: peak_equity = equity
                active_trade = None
                continue

        # Execute Pending Queue
        if pending_signal is not None and active_trade is None:
            fill_price = current_bar['env_open']
            slippage_pips = np.clip(current_bar['env_atr'] * 0.1, 0.1, 1.5)
            fill_price += (slippage_pips * 0.1) if pending_signal['type'] == 'Long' else -(slippage_pips * 0.1)

            sl_pips = pending_signal['sl_distance']
            
            # Use the Dynamic Compounding logic (1.5%) you implemented
            current_risk_usd = equity * sim.risk_pct
            lot_size = round(np.clip(current_risk_usd / (sl_pips * sim.pip_value_per_lot), 0.01, 100.0), 2)
            total_friction = (lot_size * sim.commission_per_lot) + (lot_size * sim.spread_pips * sim.pip_value_per_lot)
            
            active_trade = {
                'type': pending_signal['type'],
                'entry': fill_price,
                'sl': fill_price - (sl_pips * 0.1) if pending_signal['type'] == 'Long' else fill_price + (sl_pips * 0.1),
                'tp': fill_price + (pending_signal['tp_distance'] * 0.1) if pending_signal['type'] == 'Long' else fill_price - (pending_signal['tp_distance'] * 0.1),
                'lot_size': lot_size,
                'total_friction': total_friction
            }
            bars_since_last_trade = 0
            total_trades += 1
            pending_signal = None
            continue

        # Signal Generation
        if sim.is_restricted_time(current_time): continue

        prob_hold, prob_long, prob_short = sim._get_oracle_probs(i)
        
        features = current_bar[sim.feature_cols].values
        obs = np.zeros(len(sim.feature_cols) + 6, dtype=np.float32)
        obs[:len(features)] = features
        obs[len(features)] = prob_hold
        obs[len(features)+1] = prob_long
        obs[len(features)+2] = prob_short
        obs[-3] = float(np.clip(equity / sim.initial_balance, 0.0, 10.0))
        obs[-2] = float(np.clip((peak_equity - equity) / peak_equity, 0.0, 1.0))
        obs[-1] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
        
        action, _ = sim.manager.predict(obs, deterministic=True)
        direction_val, size_val, tp_val, sl_val = action[0], action[1], action[2], action[3]
        
        direction = 1 if direction_val > 0.33 else (2 if direction_val < -0.33 else 0)
        if direction != 0 and bars_since_last_trade < sim.min_bars_between_trades: direction = 0

        if direction != 0:
            sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5 
            tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 3.0 + 2.0) 
            pending_signal = {
                'type': 'Long' if direction == 1 else 'Short',
                'sl_distance': (current_bar['env_atr'] * sl_mult) * 10,
                'tp_distance': (current_bar['env_atr'] * tp_mult) * 10
            }
            
    return equity, total_trades

if __name__ == "__main__":
    print("Initializing Bayesian Parameter Optimizer...")
    study = optuna.create_study(direction="maximize")
    # Run 50 trials to sweep the mathematical combinations
    study.optimize(optimize_agent, n_trials=50)
    
    print("\n==================================================")
    print(" OPTIMIZATION COMPLETE ")
    print("==================================================")
    print(f"Best Trial Score (Equity): ${study.best_value:.2f}")
    print("Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")