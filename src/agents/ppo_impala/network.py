import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    """IMPALA residual block: pre-activation ReLU -> 3x3 conv, twice, plus a skip.

    Channels are preserved, so the identity skip connection adds directly.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class ConvSequence(nn.Module):
    """One IMPALA stage: 3x3 conv -> max-pool (downsample 2x) -> 2 residual blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """DeepMind IMPALA residual network (Espeholt et al. 2018), CleanRL ppo_procgen form.

    Three conv stages with channels [16, 32, 32], each = conv + max-pool + 2 residual
    blocks, then a shared 256-unit FC feeding separate actor/critic heads.

    Input: (N, 4, 84, 84) float32 in [0, 1] (pixel norm handled in MarioToPyTorch).
    """
    def __init__(self, num_actions, in_channels=4):
        super().__init__()

        channels = [16, 32, 32]
        conv_seqs = []
        c_in = in_channels
        for c_out in channels:
            conv_seqs.append(ConvSequence(c_in, c_out))
            c_in = c_out
        self.conv_seqs = nn.ModuleList(conv_seqs)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flatten = self._conv_features(dummy).shape[1]

        self.fc_shared = nn.Linear(n_flatten, 256)

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _conv_features(self, x):
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = F.relu(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._conv_features(x)
        x = F.relu(self.fc_shared(x))

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value
