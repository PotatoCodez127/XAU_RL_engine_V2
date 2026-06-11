import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from data.wfa_pipeline import WalkForwardPipeline
from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.attention_net import TemporalAttentionOracle

class ManagerPipeline:
    def __init__(self, xau_path: str, dxy_path: str, oracle_weights_path: str = None):
        self.wfa = WalkForwardPipeline(xau_path, dxy_path, embargo_bars=200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Oracle
        self.oracle = TemporalAttentionOracle(input_dim=10, seq_len=30).to(self.device)
        if oracle_weights_path and os.path.exists(oracle_weights_path):
            self.oracle.load_state_dict(torch.load(oracle_weights_path, map_location=self.device))
        self.oracle.eval() # Freeze Oracle during Manager training

    def _precompute_oracle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the entire dataset through the Oracle once to save RL compute time.
        In reality, you would map your exact 10 features here.
        """
        print("Pre-computing Oracle pattern recognition...")
        
        # Mocking the 10 features expected by the Oracle
        features = np.random.randn(len(df), 10) 
        
        # We need to simulate the 30-period rolling window for the Oracle
        probs_list = np.zeros((len(df), 3))
        
        with torch.no_grad():
            for i in range(30, len(df)):
                window = features[i-30:i]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                logits = self.oracle(window_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list[i] = probs
                
        # Append Oracle probabilities to the DataFrame
        df['prob_hold'] = probs_list[:, 0]
        df['prob_long'] = probs_list[:, 1]
        df['prob_short'] = probs_list[:, 2]
        
        return df

    def train_wfa_split(self, split_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Trains the SAC agent on a specific Walk-Forward split."""
        
        # 1. Enqueue data through the Oracle
        enriched_train = self._precompute_oracle_features(train_df.copy())
        enriched_val = self._precompute_oracle_features(val_df.copy())
        
        # 2. Initialize Environments
        train_env = Monitor(XAUDynamicEnv(df=enriched_train))
        val_env = Monitor(XAUDynamicEnv(df=enriched_val))
        
        # 3. Configure the SAC Agent
        # We use a smaller network for the Manager because the Oracle already did the heavy lifting
        policy_kwargs = dict(net_arch=[128, 128])
        
        model = SAC(
            "MlpPolicy", 
            train_env, 
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            ent_coef='auto', # Automatically tunes exploration vs exploitation
            tensorboard_log=f"./logs/wfa_split_{split_idx}/"
        )
        
        # 4. Callback to save best model during training
        eval_callback = EvalCallback(
            val_env, 
            best_model_save_path=f"./models/manager/saved/wfa_{split_idx}/",
            log_path=f"./logs/wfa_split_{split_idx}/",
            eval_freq=5000,
            deterministic=True, 
            render=False
        )
        
        print(f"--- Starting Manager Training for Split {split_idx} ---")
        model.learn(total_timesteps=50000, callback=eval_callback)
        return model