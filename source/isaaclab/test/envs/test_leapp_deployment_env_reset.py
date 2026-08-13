# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset lifecycle contract for :class:`isaaclab.envs.LeappDeploymentEnv`."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if "leapp" not in sys.modules:
    leapp = ModuleType("leapp")
    leapp.InferenceManager = object
    sys.modules["leapp"] = leapp

from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv  # noqa: E402


def test_reset_clears_renderer_history_at_authoritative_time_before_scene_update():
    """Reset state becomes visible before renderer history is cleared and sensors update."""
    events: list[object] = []
    env = LeappDeploymentEnv.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.02),
        num_rerenders_on_reset=0,
        wait_for_textures=False,
    )
    env._step_count = 7
    env.event_manager = None
    env.command_manager = None
    env.has_rtx_sensors = False
    env.scene = SimpleNamespace(
        reset=lambda _env_ids: events.append("scene.reset"),
        write_data_to_sim=lambda: events.append("scene.write"),
        update=lambda dt: events.append(("scene.update", dt)),
    )
    env.sim = SimpleNamespace(
        device="cpu",
        forward=lambda: events.append("sim.forward"),
        get_simulation_time=lambda: 1.25,
        render_context=SimpleNamespace(
            reset_scene_state_cadence=lambda simulation_time: events.append(("renderer.reset", simulation_time))
        ),
    )
    env.inference = SimpleNamespace(reset=lambda: events.append("inference.reset"))
    env._read_inputs = lambda: events.append("read.inputs") or {"input": object()}

    env.reset()

    assert events == [
        "scene.reset",
        "scene.write",
        "sim.forward",
        ("renderer.reset", 1.25),
        ("scene.update", 0.02),
        "inference.reset",
        "read.inputs",
    ]
