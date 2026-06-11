import os
from data.wfa_pipeline import WalkForwardPipeline
from models.oracle.train_oracle import train_oracle_supervised
from models.manager.train_manager import ManagerPipeline

# 1. Define File Paths
XAU_PATH = "data/raw/XAUUSDr_M1_202602241013_202606052358.csv" # Update with actual exact filename
DXY_PATH = "data/raw/USDIndex_M1_202602031250_202606052359.csv" # Update with actual exact filename
ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"

def main():
    print("=== XAU RL Engine V2 Initialization ===")
    
    # 2. Load and Embargo Data
    print("Loading and Merging MetaTrader Data...")
    pipeline = WalkForwardPipeline(XAU_PATH, DXY_PATH, embargo_bars=200)
    pipeline.load_and_merge()
    
    # Generate splits (e.g., Train on 40,000 bars, Test/Val on 10,000 bars)
    splits = pipeline.generate_splits(train_size=40000, test_size=10000, step_size=10000)
    print(f"Generated {len(splits)} Walk-Forward Splits.")

    # 3. Walk-Forward Loop
    for idx, split in enumerate(splits):
        print(f"\n--- Processing WFA Window {idx + 1}/{len(splits)} ---")
        train_df = split['train']
        val_df = split['test'] # Embargoed out-of-sample data

        # Phase A: Train the Deep Learning Oracle (Pattern Recognition)
        print("PHASE A: Supervised Oracle Training")
        train_oracle_supervised(train_df, save_path=ORACLE_WEIGHTS, epochs=20)

        # Phase B: Train the Reinforcement Learning Manager (Risk Allocation)
        print("PHASE B: SAC Agent Training")
        manager_pipeline = ManagerPipeline(XAU_PATH, DXY_PATH, oracle_weights_path=ORACLE_WEIGHTS)
        
        # This will pre-compute the Oracle's outputs to speed up RL inference 
        # and begin training the Stable-Baselines3 model.
        manager_pipeline.train_wfa_split(idx, train_df, val_df)

if __name__ == "__main__":
    main()