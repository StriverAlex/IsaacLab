# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for sensor/renderer lifecycle coordination."""

from types import SimpleNamespace

from isaaclab.scene import InteractiveScene


def test_initialize_renderers_prepares_every_sensor_before_scene_load():
    """Sensors sharing a backend each register their renderer-owned resources."""
    backend = object()
    calls = []

    def sensor(name: str):
        return SimpleNamespace(
            cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="ovrtx")),
            prepare_renderer=lambda renderer: calls.append((name, renderer)),
        )

    scene = object.__new__(InteractiveScene)
    scene._sensors = {"front": sensor("front"), "rear": sensor("rear")}
    scene.sim = SimpleNamespace(
        render_context=SimpleNamespace(get_renderer=lambda _cfg: backend),
    )

    backends = scene.initialize_renderers()

    assert backends == [backend]
    assert calls == [("front", backend), ("rear", backend)]
