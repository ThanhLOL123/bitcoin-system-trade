import pytest
import torch
from src.ml_pipeline.models.lstm_model import LSTMModel

def test_lstm_model_forward_pass():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    output_size = 1
    batch_size = 32
    sequence_length = 5

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    input_tensor = torch.randn(batch_size, sequence_length, input_size)
    output = model(input_tensor)

    assert output.shape == (batch_size, output_size)
