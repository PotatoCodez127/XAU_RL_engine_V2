import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class XAUDynamicEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(XAUDynamicEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # --- NEW ARCHITECTURAL CONSTANTS ---
        self.friction_cost = 10.0      # Simulates spread/commission
        self.oracle_threshold = 0.36   # Calibrated to 0.36 via sweep for ~1-2 trades/day
        self.cooldown_duration = 24    # 6 hours on 15m timeframe
        # -----------------------------------
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        exclude_cols = ['target', 'time', 'datetime', 'date']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('env_')]
        
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(len(self.feature_cols) + 2,), dtype=np.float32)
        
        self.current_step = 30 
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.state = np.zeros(self.observation_space.shape[0])
        
        # Initialize timer
        self.cooldown_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.cooldown_timer = 0 
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        features = self.df.loc[self.current_step, self.feature_cols].values
        obs[:len(features)] = features
        
        equity_ratio = float(np.clip(self.balance / self.initial_balance, 0.0, 10.0))
        peak_equity = max(self.initial_balance, self.peak_balance)
        self.peak_balance = max(peak_equity, self.balance)
        drawdown = float(np.clip((self.peak_balance - self.balance) / self.peak_balance, 0.0, 1.0))
        
        obs[-2] = equity_ratio
        obs[-1] = drawdown
        
        return np.clip(obs, -10.0, 10.0)
        
    def step(self, action):
        direction_val = action[0]
        size_val = action[1]
        tp_val = action[2]
        sl_val = action[3]
        
        direction = 0
        if direction_val > 0.33: direction = 1
        elif direction_val < -0.33: direction = 2
        
        # --- MECHANISM 2: Algorithmic Cooldown ---
        if self.cooldown_timer > 0:
            direction = 0
            self.cooldown_timer -= 1

        # --- MECHANISM 3: Oracle Confidence Gate (Event Trigger) ---
        prob_long = self.df.loc[self.current_step, 'prob_long']
        prob_short = self.df.loc[self.current_step, 'prob_short']
        
        if direction == 1 and prob_long < self.oracle_threshold:
            direction = 0
        elif direction == 2 and prob_short < self.oracle_threshold:
            direction = 0
            
        simulated_pnl = 0.0
        tp_mult_used = 0.0
        sl_mult_used = 0.0
        
        if direction != 0:
            risk_pct = ((size_val + 1.0) / 2.0) * 0.05
            tp_mult_used = ((tp_val + 1.0) / 2.0) * 4.0 + 1.0
            sl_mult_used = ((sl_val + 1.0) / 2.0) * 1.5 + 0.5
            amount_at_risk = self.balance * risk_pct
            
            prob_win = prob_long if direction == 1 else prob_short
            
            if np.random.rand() < prob_win:
                simulated_pnl = amount_at_risk * (tp_mult_used / sl_mult_used)
            else:
                simulated_pnl = -amount_at_risk
                
            # --- MECHANISM 1: Friction Injection ---
            simulated_pnl -= self.friction_cost
            
            # Activate Cooldown
            self.cooldown_timer = self.cooldown_duration
                
        self.balance += simulated_pnl
        
        peak_equity = max(self.initial_balance, self.peak_balance)
        self.peak_balance = max(peak_equity, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        
        raw_reward = simulated_pnl - (drawdown * self.initial_balance * 0.1)
        reward = raw_reward / (self.initial_balance * 0.01)
        
        # NOTE: The -0.05 Inactivity Penalty has been completely removed to establish a Neutral-Hold state.
            
        reward = float(np.clip(reward, -10.0, 10.0))
        
        self.current_step += 1
        terminated = self.balance < (self.initial_balance * 0.1)
        truncated = self.current_step >= len(self.df) - 1
        
        info = {
            'balance': self.balance,
            'drawdown': drawdown,
            'tp_mult_used': tp_mult_used,
            'sl_mult_used': sl_mult_used,
            'cooldown_active': self.cooldown_timer > 0,
            'prob_long': prob_long,
            'prob_short': prob_short
        }
        
        return self._get_obs(), reward, terminated, truncated, info