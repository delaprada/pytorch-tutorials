import torch
import torch.nn as nn
import torch.nn.functional as F

# option 1: create nn module
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__(self, input_size, hidden_size)
    self.linear1 = nn.Linear
