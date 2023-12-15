import torch
from torch import nn
import torch.nn.functional as F

class MuSigmaNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.mu_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.mu_sigma(x)
        mu, log_sigma_sq = torch.chunk(x, 2, dim=-1)
        sigma = torch.exp(0.5 * log_sigma_sq)

        return mu, sigma
