from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class enforcing a strict contract for all agents.
    The training loop will ONLY interact with these methods.
    """
    def __init__(self, env, **kwargs):
        self.env = env
        self.steps = 0
        self.total_timesteps = kwargs.get('total_timesteps', 1_000_000)

    @classmethod
    @abstractmethod
    def get_wrappers(cls, verbose=False):
        """
        Returns a list of wrapper classes/functions that this agent requires.
        The train loop will apply these to the environment sequentially.
        Must be a class method so the train loop can wrap envs before instantiating the agent.
        The optional ``verbose`` flag allows implementations to emit additional logging or
        select wrappers conditionally without changing the public interface.
        """
        pass

    @abstractmethod
    def action_select(self, state):
        """
        Given an observation, returns an action.
        Must handle both unbatched (testing) and batched (training) observations.
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Receives a transition from the environment.
        The agent is responsible for storing it and executing learning updates if ready.
        """
        pass

    def get_custom_metrics(self):
        """
        Returns an optional dictionary of custom metrics to log to WandB at the end of each rollout/update cycle.
        """
        return {}

    @abstractmethod
    def save_checkpoint(self, filepath, step):
        """Saves the agent state."""
        pass

    @abstractmethod
    def load_checkpoint(self, filepath):
        """Loads the agent state."""
        pass
