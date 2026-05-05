"""Atari env factory shared by data collection and evaluation.

Single source of truth for the preprocessing pipeline so collector and
evaluator can't drift. Standard recipe:

    gym.make(env, frameskip=1)               # disable env-internal frameskip
    AtariPreprocessing(frame_skip=4, screen_size=84, grayscale_obs=True)
    FrameStackObservation(stack_size=4)

`-v5` envs add sticky-action stochasticity (repeat_action_probability=0.25).
"""
from __future__ import annotations

import ale_py
import gymnasium as gym

try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    _STACK = "FrameStackObservation"
except ImportError:  # older gymnasium
    from gymnasium.wrappers import AtariPreprocessing
    from gymnasium.wrappers import FrameStack as FrameStackObservation

    _STACK = "FrameStack"

gym.register_envs(ale_py)


def make_env(
    env_name: str,
    seed: int = 0,
    frame_skip: int = 4,
    screen_size: int = 84,
    stack_size: int = 4,
    repeat_action_probability: float = 0.25,
    full_action_space: bool = False,
    noop_max: int = 30,
) -> gym.Env:
    """Build an Atari env with the standard preprocessing + frame stack."""
    env = gym.make(
        env_name,
        frameskip=1,  # -v5 default is 4; disable so AtariPreprocessing owns frame_skip
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
    )
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    if _STACK == "FrameStackObservation":
        env = FrameStackObservation(env, stack_size=stack_size)
    else:
        env = FrameStackObservation(env, num_stack=stack_size)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
