import torch
import torch.nn.functional as F
from models.oracle.custom_loss import FocalLoss

def test_focal_loss_vs_cross_entropy():
    """
    Ensures Focal Loss produces a strictly smaller loss than Cross Entropy 
    for predictions that the model is already confident and correct about.
    """
    # Simulate a batch of 3 predictions across 3 classes (Hold, Long, Short)
    # The model is VERY confident and correct about the first two, and wrong about the third.
    logits = torch.tensor([
        [5.0, -1.0, -1.0],  # Confident Class 0 (Hold)
        [-1.0, 5.0, -1.0],  # Confident Class 1 (Long)
        [-1.0, -1.0, 5.0]   # Predicted Class 2, but actual is Class 0
    ])
    
    targets = torch.tensor([0, 1, 0])

    focal_criterion = FocalLoss(gamma=2.0, reduction='none')
    
    ce_loss_raw = F.cross_entropy(logits, targets, reduction='none')
    focal_loss_raw = focal_criterion(logits, targets)

    # Assert that the Focal Loss is smaller than CE Loss for the highly confident predictions
    assert focal_loss_raw[0] < ce_loss_raw[0], "Focal loss should reduce penalty for easy sample 0"
    assert focal_loss_raw[1] < ce_loss_raw[1], "Focal loss should reduce penalty for easy sample 1"
    
    # The third sample is completely wrong. Focal loss should still be relatively high here, 
    # forcing the optimizer to pay attention to it.
    assert focal_loss_raw[2] > 0.5, "Focal loss must still penalize entirely incorrect predictions"