import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """Transformer model for Bitcoin price prediction"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.input_projection = nn.Linear(input_size, d_model)

    def forward(self, x):
        x = self.input_projection(x)
        out = self.transformer_encoder(x)
        out = self.fc(out[:, -1, :])
        return out
