import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 7 * 7, 256)

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        # orthogonal init from CleanRL layer_init()
        # gains: sqrt(2) conv/fc, 0.01 actor, 1.0 critic
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_shared.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.fc_shared.bias, 0)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc_shared(x))

        logits = self.actor(x)
        value = self.critic(x)

        return logits, value

    def get_action_and_value(self, x, action=None):
        # CleanRL Agent.get_action_and_value()
        # pixel norm handled in MarioToPyTorch
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value
