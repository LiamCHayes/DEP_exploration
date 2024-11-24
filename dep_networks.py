"""
A neural network with DEP layers
"""

import torch
import torch.nn as nn

from DEP import DEP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Single DEP layer
class DEPLayer(DEP):
    def __init__(self, input_size, output_size):
        super(DEPLayer, self).__init__(10, 1000, 0.002, 1, 1, device, output_size, input_size)

# Network where DEP is part of the input
class FirstLayerDEP(nn.Module):
    def __init__(self, input_size):
        super(FirstLayerDEP, self).__init__()

        # Layers
        self.DEPlayer = DEPLayer(input_size, 10)

    def forward(self):
        pass
