"""Shared pytest fixtures.

Single source of truth for the set of agent classes under test, so the PPO-core
and checkpoint suites run against every agent that implements the BaseAgent contract.
"""
import pytest

from src.agents.ppo_nature.agent import PPONatureAgent
from src.agents.ppo_impala.agent import PPOImpalaAgent

AGENT_CLASSES = [PPONatureAgent, PPOImpalaAgent]


@pytest.fixture(params=AGENT_CLASSES, ids=lambda c: c.__name__)
def agent_cls(request):
    """Each test using this fixture runs once per agent class."""
    return request.param
