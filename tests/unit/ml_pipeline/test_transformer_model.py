import pytest
import torch
from src.ml_pipeline.models.transformer_model import TransformerModel

def test_transformer_model_forward_pass():
    input_size = 10
    d_model = 20
    nhead = 2
    num_layers = 2
    output_size = 1
    batch_size = 32
    sequence_length = 5

    model = TransformerModel(input_size, d_model, nhead, num_layers, output_size)
    input_tensor = torch.randn(batch_size, sequence_length, input_size)
    output = model(input_tensor)

    assert output.shape == (batch_size, output_size)
