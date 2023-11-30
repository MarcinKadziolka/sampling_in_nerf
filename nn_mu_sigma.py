import torch
from torch import nn
import torch.nn.functional as F

class MuSigmaNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.mu_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.mu_sigma(x)
        # Split the output into mu and log(sigma^2)
        mu, log_sigma_sq = torch.chunk(x, 2, dim=-1)
        # Sigma should be positive, so take the exponential of log(sigma^2)
        sigma = torch.exp(0.5 * log_sigma_sq)

        return mu, sigma
