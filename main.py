import os
from data.wfa_pipeline import WalkForwardPipeline
from models.oracle.train_oracle import train_oracle_supervised
from models.manager.train_manager import ManagerPipeline

FEATURES_PATH = "data/processed/labeled_features_15m.csv" 
ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"

# --- THE CONTINUOUS TRAINING CONTROLS ---
START_SPLIT = 0 
END_SPLIT = 56

# If resuming after a crash, point this to the last successfully completed zip file
# Example: "models/manager/saved/wfa_14/best_model.zip"
RESUME_SAC_PATH = None 

def main():
    print("=== XAU RL Engine V2: Master Brain Initialization ===")
    
    pipeline = WalkForwardPipeline(FEATURES_PATH, embargo_bars=50)
    pipeline.load_data()
    
    splits = pipeline.generate_splits(train_size=10000, test_size=2500, step_size=2500)
    print(f"Generated {len(splits)} Walk-Forward Splits.")

    current_sac_path = RESUME_SAC_PATH

    for idx, split in enumerate(splits):
        if idx < START_SPLIT:
            continue
            
        if idx >= END_SPLIT:
            print(f"Reached END_SPLIT target ({END_SPLIT}). Clean Shutdown.")
            break

        print(f"\n--- Processing WFA Window {idx + 1}/{len(splits)} ---")
        train_df = split['train']
        val_df = split['test'] 

        print("PHASE A: Supervised Oracle Training")
        train_oracle_supervised(train_df, save_path=ORACLE_WEIGHTS, epochs=20)

        print("PHASE B: SAC Agent Training")
        manager_pipeline = ManagerPipeline(FEATURES_PATH, dxy_path="", oracle_weights_path=ORACLE_WEIGHTS)
        
        # Train and capture the path to the newly saved weights
        current_sac_path = manager_pipeline.train_wfa_split(
            split_idx=idx, 
            train_df=train_df, 
            val_df=val_df, 
            previous_sac_path=current_sac_path
        )

if __name__ == "__main__":
    main()