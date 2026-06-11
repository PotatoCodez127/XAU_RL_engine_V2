import pytest
import torch
import torch.optim as optim
from models.oracle.attention_net import TemporalAttentionOracle
from models.oracle.custom_loss import FocalLoss

def test_oracle_forward_pass():
    """Ensures the Attention network outputs the correct shape for classification."""
    batch_size = 16
    seq_len = 30
    input_dim = 15 # Simulated features (price, zones, dxy, etc.)
    
    # Create dummy market data
    dummy_data = torch.randn(batch_size, seq_len, input_dim)
    
    model = TemporalAttentionOracle(input_dim=input_dim, seq_len=seq_len)
    
    # Forward pass
    logits = model(dummy_data)
    
    # The output should be (batch_size, 3) for [Hold, Long, Short]
    assert logits.shape == (batch_size, 3)

def test_oracle_backward_pass():
    """Ensures the network can successfully calculate gradients and take an Adam step."""
    model = TemporalAttentionOracle(input_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(gamma=2.0)
    
    dummy_data = torch.randn(8, 30, 10)
    # Random target classes (0, 1, or 2)
    dummy_targets = torch.randint(0, 3, (8,)) 
    
    # Forward pass
    optimizer.zero_grad()
    logits = model(dummy_data)
    
    # Calculate loss
    loss = criterion(logits, dummy_targets)
    
    # Backward pass
    loss.backward()
    
    # Ensure gradients have propagated to the initial projection layer
    assert model.input_projection.weight.grad is not None
    
    # Take an optimization step
    optimizer.step()