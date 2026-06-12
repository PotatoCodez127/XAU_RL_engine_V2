import os
from data.wfa_pipeline import WalkForwardPipeline
from models.oracle.train_oracle import train_oracle_supervised
from models.manager.train_manager import ManagerPipeline

# Point directly to the engineered 15m features
FEATURES_PATH = "data/processed/labeled_features_15m.csv" 
ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"

def main():
    print("=== XAU RL Engine V2 Initialization ===")
    
    pipeline = WalkForwardPipeline(FEATURES_PATH, embargo_bars=50)
    pipeline.load_data()
    
    # Adjusted for 15m Timeframe:
    # Train Size: 10,000 bars (~4 months)
    # Test Size: 2,500 bars (~1 month out of sample)
    splits = pipeline.generate_splits(train_size=10000, test_size=2500, step_size=2500)
    print(f"Generated {len(splits)} Walk-Forward Splits.")

    for idx, split in enumerate(splits):
        print(f"\n--- Processing WFA Window {idx + 1}/{len(splits)} ---")
        train_df = split['train']
        val_df = split['test'] 

        print("PHASE A: Supervised Oracle Training")
        train_oracle_supervised(train_df, save_path=ORACLE_WEIGHTS, epochs=20)

        print("PHASE B: SAC Agent Training")
        # Initialize the Manager pipeline, note we don't need DXY path anymore 
        # because it's already merged in the master features!
        manager_pipeline = ManagerPipeline(FEATURES_PATH, dxy_path="", oracle_weights_path=ORACLE_WEIGHTS)
        manager_pipeline.train_wfa_split(idx, train_df, val_df)

if __name__ == "__main__":
    main()