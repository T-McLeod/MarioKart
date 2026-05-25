from ..base import BaseAgent
from .network import ImpalaCNN

class PPOImpalaAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # TODO: Initialize IMPALA network, optimizer, and buffers
        raise NotImplementedError("PPOImpalaAgent initialization is not implemented yet.")

    @classmethod
    def get_wrappers(cls, verbose=False):
        # TODO: Define wrappers specific to IMPALA (e.g. RGB input instead of Grayscale)
        return []

    def action_select(self, state):
        # TODO: Implement action selection
        pass

    def update(self, state, action, reward, next_state, done):
        # TODO: Implement rollout storage and PPO update trigger
        pass

    def save_checkpoint(self, filepath, step):
        pass

    def load_checkpoint(self, filepath):
        pass
