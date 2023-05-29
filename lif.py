import torch
import torch.nn as nn

class LIFLayer(nn.Module):
    def __init__(self, weights, dt=1e-2, threshold=1.0, tau_v=1e-2, tau_inhibitory=1e-2, inhibitory_scale=0.1):
        super(LIFLayer, self).__init__()
        self.weights = weights
        self.dt = dt
        self.threshold = threshold
        self.tau_v = tau_v
        self.inhibitory_scale = inhibitory_scale
        self.tau_inhibitory = tau_inhibitory

        self.state = torch.zeros(weights.shape[0])
        self.mean_activation = torch.zeros(1)

    def forward(self, x):
        # Calculate v
        x = x * self.weights
        dv_dt = -self.state / self.tau_v + x - self.mean_activation
        self.state += dv_dt * self.dt

        # Check for spiking activity
        spikes = self.state > self.threshold

        # Update astrocyte state
        d_inhibition = -self.mean_activation * self.dt / self.tau_inhibitory
        self.mean_activation += d_inhibition + torch.sum(spikes.int()) * self.inhibitory_scale

        # clamp
        self.state[spikes] = 0
        self.state[self.state < 0] = 0

        return spikes
