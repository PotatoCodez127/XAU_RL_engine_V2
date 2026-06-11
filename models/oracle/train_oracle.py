import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from models.oracle.attention_net import TemporalAttentionOracle
from models.oracle.custom_loss import FocalLoss

def train_oracle_supervised(df: pd.DataFrame, save_path: str, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Oracle on {device}...")

    # Initialize Model & Optimizer
    # Assuming 10 processed features (price action, 4H/30m zone distances, DXY correlation)
    model = TemporalAttentionOracle(input_dim=10, seq_len=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=2.0)

    # Note: In reality, you will extract your 10 features and your Target labels here.
    # For demonstration, we simulate the processed feature tensors:
    features = torch.randn(len(df) - 30, 30, 10).to(device)
    targets = torch.randint(0, 3, (len(df) - 30,)).to(device) # 0: Hold, 1: Long, 2: Short

    # Standard PyTorch Training Loop
    model.train()
    batch_size = 256
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(features), batch_size):
            batch_x = features[i:i+batch_size]
            batch_y = targets[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Focal Loss: {epoch_loss/len(features):.4f}")

    # Save the 'Brain'
    torch.save(model.state_dict(), save_path)
    print(f"Oracle weights saved to {save_path}")
    return model