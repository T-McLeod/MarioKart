class MarioKartRandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        return self.env.action_space.sample()
    

    def update(self, state, action, reward, next_state, terminated):
        pass  # No learning for a random agent