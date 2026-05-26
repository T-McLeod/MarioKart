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
        raise NotImplementedError(
            "PPOImpalaAgent.get_wrappers() is not implemented yet."
        )

    def action_select(self, state):
        # TODO: Implement action selection
        raise NotImplementedError(
            "PPOImpalaAgent.action_select() is not implemented yet."
        )

    def update(self, state, action, reward, next_state, done):
        # TODO: Implement rollout storage and PPO update trigger
        raise NotImplementedError(
            "PPOImpalaAgent.update() is not implemented yet."
        )

    def save_checkpoint(self, filepath, step):
        raise NotImplementedError(
            "PPOImpalaAgent.save_checkpoint() is not implemented yet."
        )

    def load_checkpoint(self, filepath):
        raise NotImplementedError(
            "PPOImpalaAgent.load_checkpoint() is not implemented yet."
        )
