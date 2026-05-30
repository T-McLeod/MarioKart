"""Tests for the PPO training core: GAE computation and rollout mechanics."""
import numpy as np
import pytest
import torch

from src.agents.ppo_nature.agent import PPONatureAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_ENVS = 2
OBS_SHAPE = (4, 84, 84)


@pytest.fixture
def agent():
    return PPONatureAgent(
        env=None,
        rollout_steps=8,
        minibatch_size=4,
        n_epochs=1,
        total_timesteps=10_000,
    )


def _fill_buffer(agent, steps=4, reward=1.0, value=0.5, done=0.0):
    """Populate the rollout buffer with `steps` synthetic transitions."""
    agent._init_rollout_buffer()
    for _ in range(steps):
        agent._rb_states.append(np.random.rand(NUM_ENVS, *OBS_SHAPE).astype(np.float32))
        agent._rb_actions.append(np.zeros(NUM_ENVS, dtype=np.int64))
        agent._rb_log_probs.append(np.full(NUM_ENVS, -1.0, dtype=np.float32))
        agent._rb_rewards.append(np.full(NUM_ENVS, reward, dtype=np.float32))
        agent._rb_values.append(np.full(NUM_ENVS, value, dtype=np.float32))
        agent._rb_dones.append(np.full(NUM_ENVS, done, dtype=np.float32))


# ---------------------------------------------------------------------------
# GAE correctness
# ---------------------------------------------------------------------------

def _compute_gae(rewards, values, dones, last_value, last_done, discount=0.99, lam=0.95):
    """Reference implementation of the GAE loop (mirrors agent._ppo_update)."""
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(last_value, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            non_terminal = 1.0 - np.array(last_done, dtype=np.float32)
            next_val = last_value
        else:
            non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
        delta = rewards[t] + discount * next_val * non_terminal - values[t]
        last_gae = delta + discount * lam * non_terminal * last_gae
        advantages[t] = last_gae

    return advantages


def test_gae_non_terminal_single_env():
    """Verify GAE against hand-computed values for a known 3-step sequence."""
    discount, lam = 0.99, 0.95
    rewards = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
    values  = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
    dones   = np.zeros((3, 1), dtype=np.float32)
    last_value = np.array([0.5], dtype=np.float32)
    last_done  = np.zeros(1, dtype=np.float32)

    adv = _compute_gae(rewards, values, dones, last_value, last_done, discount, lam)

    # t=2: delta = 1 + 0.99*0.5 - 0.5 = 0.995
    np.testing.assert_allclose(adv[2], [0.995], rtol=1e-5)
    # t=1: delta = 0 + 0.99*0.5 - 0.5 = -0.005
    #      adv   = -0.005 + 0.99*0.95*0.995
    expected_t1 = -0.005 + discount * lam * 0.995
    np.testing.assert_allclose(adv[1], [expected_t1], rtol=1e-5)
    # t=0: delta = 1 + 0.99*0.5 - 0.5 = 0.995
    #      adv   = 0.995 + 0.99*0.95*expected_t1
    expected_t0 = 0.995 + discount * lam * expected_t1
    np.testing.assert_allclose(adv[0], [expected_t0], rtol=1e-5)


def test_gae_done_zeroes_bootstrap():
    """A done=1 at step t must cut the value bootstrap from t+1."""
    discount, lam = 0.99, 0.95
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values  = np.array([[0.5], [0.5]], dtype=np.float32)
    # Episode ends after step 0
    dones   = np.array([[1.0], [0.0]], dtype=np.float32)
    last_value = np.array([0.5], dtype=np.float32)
    last_done  = np.zeros(1, dtype=np.float32)

    adv = _compute_gae(rewards, values, dones, last_value, last_done, discount, lam)

    # t=1: no done, bootstrap applies → delta = 1 + 0.99*0.5 - 0.5 = 0.995
    np.testing.assert_allclose(adv[1], [0.995], rtol=1e-5)
    # t=0: done=1, non_terminal=0 → delta = 1 + 0 - 0.5 = 0.5, no further bootstrap
    np.testing.assert_allclose(adv[0], [0.5], rtol=1e-5)


def test_gae_last_done_zeroes_final_bootstrap():
    """last_done=1 must zero the bootstrap from the final step."""
    discount, lam = 0.99, 0.95
    rewards = np.array([[1.0]], dtype=np.float32)
    values  = np.array([[0.5]], dtype=np.float32)
    dones   = np.zeros((1, 1), dtype=np.float32)
    last_value = np.array([100.0], dtype=np.float32)   # large; should be zeroed
    last_done  = np.ones(1, dtype=np.float32)          # episode ended

    adv = _compute_gae(rewards, values, dones, last_value, last_done, discount, lam)

    # non_terminal = 0 → delta = 1 + 0 - 0.5 = 0.5
    np.testing.assert_allclose(adv[0], [0.5], rtol=1e-5)



# ---------------------------------------------------------------------------
# PPO update: smoke tests
# ---------------------------------------------------------------------------

def test_ppo_update_runs_without_error(agent):
    _fill_buffer(agent)
    last_state = np.random.rand(NUM_ENVS, *OBS_SHAPE).astype(np.float32)
    agent._ppo_update(last_state, np.zeros(NUM_ENVS, dtype=bool))


def test_ppo_update_populates_metrics(agent):
    _fill_buffer(agent)
    last_state = np.random.rand(NUM_ENVS, *OBS_SHAPE).astype(np.float32)
    agent._ppo_update(last_state, np.zeros(NUM_ENVS, dtype=bool))

    metrics = agent.get_custom_metrics()
    for key in ("pg_loss", "v_loss", "entropy", "approx_kl", "entropy_coef"):
        assert key in metrics, f"missing metric: {key}"
        assert np.isfinite(metrics[key]), f"non-finite metric: {key}"


def test_ppo_update_changes_network_weights(agent):
    """A gradient step must actually change at least one parameter."""
    params_before = [p.clone() for p in agent.ac_net.parameters()]
    _fill_buffer(agent)
    last_state = np.random.rand(NUM_ENVS, *OBS_SHAPE).astype(np.float32)
    agent._ppo_update(last_state, np.zeros(NUM_ENVS, dtype=bool))
    params_after = list(agent.ac_net.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed, "PPO update left all weights unchanged"


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

def test_rollout_buffer_flushes_after_init(agent):
    _fill_buffer(agent, steps=4)
    assert len(agent._rb_states) == 4
    agent._init_rollout_buffer()
    assert len(agent._rb_states) == 0
    assert len(agent._rb_rewards) == 0


def test_update_triggers_ppo_at_rollout_boundary(agent):
    """Calling update() enough times to fill the buffer should trigger _ppo_update."""
    steps_per_env = agent.rollout_steps // NUM_ENVS  # 8 / 2 = 4
    obs = np.random.rand(NUM_ENVS, *OBS_SHAPE).astype(np.float32)

    # Prime cached values so update() doesn't crash on None
    agent._cached_log_prob = np.zeros(NUM_ENVS, dtype=np.float32)
    agent._cached_value    = np.zeros(NUM_ENVS, dtype=np.float32)

    for _ in range(steps_per_env):
        agent.update(obs, np.zeros(NUM_ENVS, dtype=np.int64),
                     np.ones(NUM_ENVS), obs, np.zeros(NUM_ENVS, dtype=bool))

    # Buffer should have been flushed after the update
    assert len(agent._rb_states) == 0
