import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

class HighFidelitySimulator:
    def __init__(self, data_path, oracle_path, manager_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing High-Fidelity V3 Simulator...")
        
        # Load Data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Broker & Risk Constants
        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0 # Standard $10 per pip on 1 standard XAUUSD lot (100oz)
        self.risk_usd = 100.0
        self.initial_balance = 10000.0
        
        # System Constants
        self.oracle_threshold = 0.36
        self.cooldown_duration = 24
        
        # Load AI Models
        self._load_models(oracle_path, manager_path)

    def _load_models(self, oracle_path, manager_path):
        exclude_cols = ['target', 'time', 'datetime', 'date']
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols and not c.startswith('env_')]
        
        # Phase A: Oracle
        self.oracle = TemporalAttentionOracle(input_dim=len(self.feature_cols), seq_len=30).to(self.device)
        self.oracle.load_state_dict(torch.load(oracle_path, map_location=self.device))
        self.oracle.eval()
        
        # Phase B: Manager (Load purely for inference, no environment needed)
        self.manager = SAC.load(manager_path, device=self.device)

    def _get_oracle_probs(self, current_step):
        # Retrieve the 30-period rolling window
        window = self.df[self.feature_cols].iloc[current_step-30:current_step].values
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        return probs[1], probs[2] # prob_long, prob_short

    def is_restricted_time(self, current_time: pd.Timestamp) -> bool:
        """Enforces real-world liquidity voids."""
        # 1. Daily Rollover (23:45 to 00:30)
        if (current_time.hour == 23 and current_time.minute >= 45) or \
           (current_time.hour == 0 and current_time.minute <= 30):
            return True
        # 2. Friday Close (No entries after 21:00 Friday)
        if current_time.weekday() == 4 and current_time.hour >= 21:
            return True
        return False

    def run_simulation(self):
        print("Executing Asynchronous Backtest Engine...")
        
        equity = self.initial_balance
        peak_equity = equity
        equity_curve = [equity]
        journal = []
        
        cooldown_timer = 0
        active_trade = None
        pending_signal = None
        
        # Start at step 30 to allow for the initial Oracle window
        for i in range(30, len(self.df) - 1):
            current_time = self.df.index[i]
            current_bar = self.df.iloc[i]
            next_bar = self.df.iloc[i+1] # Lookahead strictly for execution latency
            
            # --- 1. NON-BLOCKING PRICE TRACKING (Manage Active Trade) ---
            if active_trade is not None:
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""
                
                # Friday forced liquidation check
                if current_time.weekday() == 4 and current_time.hour >= 22 and current_time.minute >= 45:
                    trade_closed = True
                    exit_price = current_bar['env_close']
                    exit_reason = "Weekend Liquidation"
                else:
                    if active_trade['type'] == 'Long':
                        if current_bar['env_low'] <= active_trade['sl']:
                            trade_closed, exit_price, exit_reason = True, active_trade['sl'], "Stop Loss"
                        elif current_bar['env_high'] >= active_trade['tp']:
                            trade_closed, exit_price, exit_reason = True, active_trade['tp'], "Take Profit"
                    
                    elif active_trade['type'] == 'Short':
                        if current_bar['env_high'] >= active_trade['sl']:
                            trade_closed, exit_price, exit_reason = True, active_trade['sl'], "Stop Loss"
                        elif current_bar['env_low'] <= active_trade['tp']:
                            trade_closed, exit_price, exit_reason = True, active_trade['tp'], "Take Profit"

                if trade_closed:
                    # Calculate exact PnL
                    pip_diff = (exit_price - active_trade['entry']) * 10 # Assuming XAU 0.1 = 1 pip
                    if active_trade['type'] == 'Short': pip_diff *= -1
                    
                    gross_pnl = pip_diff * self.pip_value_per_lot * active_trade['lot_size']
                    net_pnl = gross_pnl - active_trade['total_friction']
                    
                    equity += net_pnl
                    if equity > peak_equity: peak_equity = equity
                    
                    journal.append({
                        'Entry_Time': active_trade['time'], 'Exit_Time': current_time,
                        'Type': active_trade['type'], 'Lot_Size': round(active_trade['lot_size'], 2),
                        'Friction_Cost': round(active_trade['total_friction'], 2),
                        'Net_PnL': round(net_pnl, 2), 'Equity': round(equity, 2), 'Reason': exit_reason
                    })
                    
                    active_trade = None
                    cooldown_timer = self.cooldown_duration
                    continue # Skip to next candle to enforce separation of execution

            # --- 2. EXECUTE PENDING QUEUE (Latency Simulation $t$) ---
            if pending_signal is not None and active_trade is None:
                # We execute at the OPEN of the current bar, simulating the delay from previous CLOSE
                fill_price = current_bar['env_open']
                
                # Dynamic ATR Slippage
                atr = current_bar['env_atr']
                slippage_pips = np.clip(atr * 0.1, 0.1, 1.5) # Scale slippage based on volatility
                
                if pending_signal['type'] == 'Long':
                    fill_price += (slippage_pips * 0.1)
                else:
                    fill_price -= (slippage_pips * 0.1)

                # Volumetric Math
                sl_pips = pending_signal['sl_distance']
                # Lot_Volume = Risk_USD / (SL_Pips * Pip_Value)
                lot_size = self.risk_usd / (sl_pips * self.pip_value_per_lot)
                lot_size = round(np.clip(lot_size, 0.01, 100.0), 2)
                
                # Friction
                commission = lot_size * self.commission_per_lot
                spread_cost = lot_size * self.spread_pips * self.pip_value_per_lot
                total_friction = commission + spread_cost
                
                active_trade = {
                    'time': current_time,
                    'type': pending_signal['type'],
                    'entry': fill_price,
                    'sl': fill_price - (sl_pips * 0.1) if pending_signal['type'] == 'Long' else fill_price + (sl_pips * 0.1),
                    'tp': fill_price + (pending_signal['tp_distance'] * 0.1) if pending_signal['type'] == 'Long' else fill_price - (pending_signal['tp_distance'] * 0.1),
                    'lot_size': lot_size,
                    'total_friction': total_friction
                }
                pending_signal = None
                continue

            # --- 3. SIGNAL GENERATION (The Oracle / Manager) ---
            if cooldown_timer > 0:
                cooldown_timer -= 1
                equity_curve.append(equity)
                continue

            if self.is_restricted_time(current_time):
                equity_curve.append(equity)
                continue

            prob_long, prob_short = self._get_oracle_probs(i)
            
            # Construct standard observation vector for the SAC Agent
            features = current_bar[self.feature_cols].values
            obs = np.zeros(len(self.feature_cols) + 2, dtype=np.float32)
            obs[:len(features)] = features
            obs[-2] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
            obs[-1] = float(np.clip((peak_equity - equity) / peak_equity, 0.0, 1.0))
            
            action, _ = self.manager.predict(obs, deterministic=True)
            direction_val, size_val, tp_val, sl_val = action[0], action[1], action[2], action[3]
            
            direction = 0
            if direction_val > 0.33 and prob_long >= self.oracle_threshold: direction = 1
            elif direction_val < -0.33 and prob_short >= self.oracle_threshold: direction = 2

            if direction != 0:
                # Queue the order for the NEXT candle ($t$ delay)
                sl_mult = ((sl_val + 1.0) / 2.0) * 1.5 + 0.5
                tp_mult = ((tp_val + 1.0) / 2.0) * 4.0 + 1.0
                
                # Convert multipliers to absolute pips using current ATR
                sl_distance_pips = (current_bar['env_atr'] * sl_mult) * 10 
                tp_distance_pips = (current_bar['env_atr'] * tp_mult) * 10
                
                pending_signal = {
                    'type': 'Long' if direction == 1 else 'Short',
                    'sl_distance': sl_distance_pips,
                    'tp_distance': tp_distance_pips
                }
                
            equity_curve.append(equity)

        # Print Final Report
        journal_df = pd.DataFrame(journal)
        print("\n" + "="*50)
        print(" 📡 HIGH-FIDELITY LIVE SIMULATION REPORT 📡")
        print("="*50)
        if not journal_df.empty:
            print(f"Total Trades Executed: {len(journal_df)}")
            print(f"Final Account Equity:  ${equity:.2f}")
            print(f"Average Lot Size:      {journal_df['Lot_Size'].mean():.2f} Lots")
            print(f"Average Friction/Trade:${journal_df['Friction_Cost'].mean():.2f}")
            wins = journal_df[journal_df['Net_PnL'] > 0]
            print(f"True Winrate:          {(len(wins)/len(journal_df))*100:.2f}%")
        else:
            print("No trades executed. Thresholds or temporal voids blocked all entries.")
        print("="*50)
        
        journal_df.to_csv("logs/high_fidelity_journal.csv", index=False)
        print("Detailed execution log saved to logs/high_fidelity_journal.csv")

if __name__ == "__main__":
    DATA = "data/processed/labeled_features_15m.csv"
    ORACLE = "models/oracle/best_oracle.pth"
    MANAGER = "models/manager/saved/wfa_44/best_model.zip"
    
    sim = HighFidelitySimulator(DATA, ORACLE, MANAGER)
    sim.run_simulation()