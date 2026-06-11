import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the 
    tokens in the sequence. Attention is permutation-invariant, so it needs 
    this to understand time.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TemporalAttentionOracle(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 30, d_model: int = 64, n_heads: int = 4, num_classes: int = 3):
        """
        input_dim: Number of features (e.g., XAU close, DXY, 30m/4H zone distances)
        seq_len: The lookback window (default 30)
        num_classes: Hold (0), Long (1), Short (2)
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Self-Attention Layer
        # batch_first=True expects (batch, seq, feature)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Global Average Pooling and Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project raw features into the model dimension
        x = self.input_projection(x)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Multi-Head Self Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out) # Residual connection
        
        # 4. Feed Forward
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out) # Residual connection
        
        # 5. Global Average Pooling (collapse the sequence dimension)
        # We average across the 30 candles to get a single summary vector per batch
        x = x.mean(dim=1) 
        
        # 6. Final Classification
        logits = self.classifier(x)
        
        # Note: We return raw logits, NOT softmax probabilities. 
        # The FocalLoss/CrossEntropyLoss handles the softmax internally for better numerical stability.
        return logits