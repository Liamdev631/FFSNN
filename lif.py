import torch
import torch.nn as nn

class LIFLayer(nn.Module):
    def __init__(self, weights, device='cpu', dt=1e-3, threshold=1.0, tau_v=20e-3, tau_inhibitory=60e-3, inhibitory_scale=10, R=100):
        super(LIFLayer, self).__init__()
        self.weights = weights.to(device)
        self.dt = dt
        self.threshold = threshold
        self.R = R
        self.device = device

        self.v = torch.zeros(size=(weights.shape[0],), device=device)
        self.tau_v = tau_v

        self.activation = torch.zeros(size=(1,), device=device)
        self.tau_activation = tau_inhibitory
        self.inhibitory_scale = inhibitory_scale

    def forward(self, x):
        # Calculate v
        x = x * self.weights
        dv_dt = -(self.v / self.tau_v) + x * self.R - self.activation
        self.v += dv_dt * self.dt

        # Check for spiking activity
        spikes = self.v > self.threshold

        # Update astrocyte state
        self.activation += torch.sum(spikes).float() * self.inhibitory_scale
        self.activation -= self.activation * self.dt / self.tau_activation

        # Clamp
        self.v[spikes] = 0
        self.v[self.v < 0] = 0

        return spikes
