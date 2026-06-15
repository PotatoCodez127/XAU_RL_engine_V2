import os
from data.wfa_pipeline import WalkForwardPipeline
from models.oracle.train_oracle import train_oracle_supervised
from models.manager.train_manager import ManagerPipeline

FEATURES_PATH = "data/processed/labeled_features_15m.csv" 
ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"

# --- RESUME CONTROL ---
# Change this number to the split index where Colab disconnected.
# For example, if it died during "Processing WFA Window 15/56", set this to 14 (0-indexed).
START_SPLIT = 0 

def main():
    print("=== XAU RL Engine V2 Initialization ===")
    
    pipeline = WalkForwardPipeline(FEATURES_PATH, embargo_bars=50)
    pipeline.load_data()
    
    splits = pipeline.generate_splits(train_size=10000, test_size=2500, step_size=2500)
    print(f"Generated {len(splits)} Walk-Forward Splits.")

    for idx, split in enumerate(splits):
        # Fast-forward to the disconnected split
        if idx < START_SPLIT:
            print(f"Skipping Split {idx} (Already Completed)...")
            continue

        print(f"\n--- Processing WFA Window {idx + 1}/{len(splits)} ---")
        train_df = split['train']
        val_df = split['test'] 

        print("PHASE A: Supervised Oracle Training")
        train_oracle_supervised(train_df, save_path=ORACLE_WEIGHTS, epochs=20)

        print("PHASE B: SAC Agent Training")
        manager_pipeline = ManagerPipeline(FEATURES_PATH, dxy_path="", oracle_weights_path=ORACLE_WEIGHTS)
        manager_pipeline.train_wfa_split(idx, train_df, val_df)

if __name__ == "__main__":
    main()