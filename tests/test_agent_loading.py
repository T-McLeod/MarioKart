"""Tests for the dynamic agent-discovery mechanism used by the training loop.

Mirrors the importlib lookup in src/train.py (``f"src.agents.{args.agent}.agent"``)
without importing train.py itself, which top-level-imports stable_retro.
"""
import importlib
import inspect

import pytest

from src.agents.base import BaseAgent


@pytest.mark.parametrize("agent_name", ["ppo_nature", "ppo_impala"])
def test_agent_module_exposes_one_baseagent_subclass(agent_name):
    mod = importlib.import_module(f"src.agents.{agent_name}.agent")

    found = [
        obj
        for _, obj in inspect.getmembers(mod)
        if inspect.isclass(obj)
        and issubclass(obj, BaseAgent)
        and obj is not BaseAgent
        and obj.__module__ == mod.__name__
    ]

    assert len(found) == 1, (
        f"Expected exactly one BaseAgent subclass defined in {mod.__name__}, "
        f"found {[c.__name__ for c in found]}"
    )
