"""Tests for agent checkpoint save/load round-trip (all BaseAgent implementations)."""
import torch
import pytest


def _build(agent_cls):
    return agent_cls(
        env=None,
        rollout_steps=512,
        minibatch_size=64,
        n_epochs=1,
        total_timesteps=10_000,
    )


@pytest.fixture
def agent(agent_cls):
    return _build(agent_cls)


def test_save_load_preserves_step_count(agent, agent_cls, tmp_path):
    agent.steps = 99_999
    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=7)

    agent2 = _build(agent_cls)
    agent2.load_checkpoint(str(tmp_path / "ckpt"))
    assert agent2.steps == 99_999


def test_save_load_preserves_episode_number(agent, agent_cls, tmp_path):
    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=42)

    agent2 = _build(agent_cls)
    resumed = agent2.load_checkpoint(str(tmp_path / "ckpt"))
    assert resumed == 42


def test_save_load_preserves_network_weights(agent, agent_cls, tmp_path):
    with torch.no_grad():
        for p in agent.ac_net.parameters():
            p.fill_(0.42)

    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=1)

    agent2 = _build(agent_cls)
    agent2.load_checkpoint(str(tmp_path / "ckpt"))

    for p in agent2.ac_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.42)), \
            "Network weights not restored correctly after load"


def test_save_load_preserves_optimizer_state(agent, agent_cls, tmp_path):
    dummy = torch.zeros(1, 4, 84, 84)
    _, log_prob, entropy, value = agent.ac_net.get_action_and_value(dummy)
    loss = -log_prob.mean() + value.mean()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    agent.save_checkpoint(str(tmp_path / "ckpt"), episode=1)

    agent2 = _build(agent_cls)
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
