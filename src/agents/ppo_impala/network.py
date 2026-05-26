import torch.nn as nn

class ImpalaCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # TODO: Implement the DeepMind IMPALA residual architecture
        raise NotImplementedError(
            "ImpalaCNN is not implemented yet: initialize the IMPALA architecture "
            "before using this model."
        )

    def forward(self, x):
        raise NotImplementedError(
            "ImpalaCNN.forward is not implemented yet."
        )

    def get_action_and_value(self, x, action=None):
        raise NotImplementedError(
            "ImpalaCNN.get_action_and_value is not implemented yet."
        )
