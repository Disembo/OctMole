import torch
import torch.nn as nn


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    @torch.no_grad()
    def update_ema(self):
        for target, source in zip(self.ema_params1, self.parameters()):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1 - self.ema_decay1)
        for target, source in zip(self.ema_params2, self.parameters()):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1 - self.ema_decay2)
