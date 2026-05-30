"""Tests for PPONatureAgent checkpoint save/load round-trip."""
import torch
import pytest

from src.agents.ppo_nature.agent import PPONatureAgent


@pytest.fixture
def agent():
    return PPONatureAgent(
        env=None,
        rollout_steps=512,
        minibatch_size=64,
        n_epochs=1,
        total_timesteps=10_000,
    )


def test_save_load_preserves_step_count(agent, tmp_path):
    agent.steps = 99_999
    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=7)

    agent2 = PPONatureAgent(env=None, rollout_steps=512, minibatch_size=64,
                            n_epochs=1, total_timesteps=10_000)
    agent2.load_checkpoint(str(tmp_path / "ckpt"))
    assert agent2.steps == 99_999


def test_save_load_preserves_episode_number(agent, tmp_path):
    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=42)

    agent2 = PPONatureAgent(env=None, rollout_steps=512, minibatch_size=64,
                            n_epochs=1, total_timesteps=10_000)
    resumed = agent2.load_checkpoint(str(tmp_path / "ckpt"))
    assert resumed == 42


def test_save_load_preserves_network_weights(agent, tmp_path):
    # Pin all weights to a known value
    with torch.no_grad():
        for p in agent.ac_net.parameters():
            p.fill_(0.42)

    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=1)

    agent2 = PPONatureAgent(env=None, rollout_steps=512, minibatch_size=64,
                            n_epochs=1, total_timesteps=10_000)
    agent2.load_checkpoint(str(tmp_path / "ckpt"))

    for p in agent2.ac_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.42)), \
            "Network weights not restored correctly after load"


def test_save_load_preserves_optimizer_state(agent, tmp_path):
    # Run a dummy forward/backward to populate optimizer state (momentum buffers etc.)
    import numpy as np
    dummy = torch.zeros(1, 4, 84, 84)
    _, log_prob, entropy, value = agent.ac_net.get_action_and_value(dummy)
    loss = -log_prob.mean() + value.mean()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=1)

    agent2 = PPONatureAgent(env=None, rollout_steps=512, minibatch_size=64,
                            n_epochs=1, total_timesteps=10_000)
    agent2.load_checkpoint(str(tmp_path / "ckpt"))

    for (k1, v1), (k2, v2) in zip(
        agent.optimizer.state_dict()["state"].items(),
        agent2.optimizer.state_dict()["state"].items(),
    ):
        for buf_key in v1:
            assert torch.allclose(v1[buf_key], v2[buf_key]), \
                f"Optimizer buffer '{buf_key}' mismatch after load"


def test_save_produces_model_file(agent, tmp_path):
    base = str(tmp_path / "run" / "ckpt")
    agent.save_checkpoint(base, episode=3)
    import os
    assert os.path.exists(f"{base}_model.pth")


def test_load_nonexistent_raises(agent, tmp_path):
    with pytest.raises(FileNotFoundError):
        agent.load_checkpoint(str(tmp_path / "ghost"))
