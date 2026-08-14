# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless contracts for the OVRTX LiDAR sensor adapter."""

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_ov.sensors import OVRTXLiDAR, OVRTXLiDARData, OVRTXLiDAROutputMetadata
from isaaclab_ov.sensors.lidar import ovrtx_lidar as ovrtx_lidar_module
from isaaclab_ov.sensors.lidar.ovrtx_lidar_product import OVRTXLiDARFrame


def _frame(value: float) -> OVRTXLiDARFrame:
    return OVRTXLiDARFrame(
        tensors={
            "Coordinates": torch.full((3, 2), value),
            "Counts": torch.tensor([2], dtype=torch.int32),
            "Flags": torch.full((2,), 0x40, dtype=torch.uint8),
        },
        params={},
        metadata=OVRTXLiDAROutputMetadata(
            coordinate_frame="SENSOR",
            motion_compensated=False,
            partial_outputs=False,
            instant_lidar=True,
            frame_rate_hz=10.0,
            max_returns=1,
        ),
    )


def test_sensor_spawns_an_optional_authored_lidar_profile(monkeypatch):
    """A sensor-owned spawner hides profile authoring and replication ordering from scene callers."""
    from isaaclab_ov.sensors import OVRTXLiDARCfg

    from pxr import Usd

    from isaaclab.sensors import sensor_base as sensor_base_module
    from isaaclab.sim import SpawnerCfg
    from isaaclab.sim.utils import create_prim
    from isaaclab.sim.utils.stage import use_stage

    spawn_calls = []
    queued_cfgs = []

    class CallbackHandle:
        def deregister(self):
            return None

    class PhysicsManager:
        @classmethod
        def register_callback(cls, *_args, **_kwargs):
            return CallbackHandle()

    context = SimpleNamespace(
        physics_manager=PhysicsManager,
        vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda _sensor: None),
    )
    monkeypatch.setattr(
        sensor_base_module.sim_utils.SimulationContext,
        "instance",
        classmethod(lambda _cls: context),
    )
    monkeypatch.setattr(ovrtx_lidar_module.cloner, "queue_replication", queued_cfgs.append)

    def spawn_profile(prim_path, _cfg, translation=None, orientation=None):
        spawn_calls.append((prim_path, translation, orientation))
        return create_prim(
            prim_path,
            "OmniLidar",
            translation=translation,
            orientation=orientation,
        )

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot", "Xform")
    cfg = OVRTXLiDARCfg(
        prim_path="/World/Robot/Lidar",
        spawn=SpawnerCfg(func=spawn_profile),
        offset=OVRTXLiDARCfg.OffsetCfg(
            pos=(0.1, 0.2, 0.3),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    with use_stage(stage):
        sensor = OVRTXLiDAR(cfg)
    try:
        assert spawn_calls == [
            (
                "/World/Robot/Lidar",
                (0.1, 0.2, 0.3),
                (0.0, 0.0, 0.0, 1.0),
            )
        ]
        assert stage.GetPrimAtPath("/World/Robot/Lidar").GetTypeName() == "OmniLidar"
        assert queued_cfgs == [cfg]
    finally:
        sensor._clear_callbacks()


def test_update_reads_only_outdated_products_from_the_initialized_renderer(monkeypatch):
    """A data update consumes the prepared renderer and does not lazily initialize scene ownership."""
    events = []

    class Renderer:
        def initialize_lidar_scene(self, num_envs):
            raise AssertionError(f"renderer scene must be initialized before sensor updates: {num_envs}")

        def read_lidar(self, product_paths):
            events.append(("read", product_paths))
            if sum(event[0] == "read" for event in events) > 1:
                return {}
            return {path: _frame(float(index + 1)) for index, path in enumerate(product_paths)}

    render_context = SimpleNamespace(
        prepare_renderer_frame=lambda renderer, step, simulation_time: events.append(
            ("prepare", renderer, step, simulation_time)
        )
    )
    vis_marker_registry = SimpleNamespace(clear_debug_vis_callback=lambda _sensor: None)
    monkeypatch.setattr(
        ovrtx_lidar_module,
        "SimulationContext",
        SimpleNamespace(
            instance=lambda: SimpleNamespace(
                render_context=render_context,
                get_physics_step_count=lambda: 12,
                get_simulation_time=lambda: 0.24,
                vis_marker_registry=vis_marker_registry,
            )
        ),
    )
    sensor = OVRTXLiDAR.__new__(OVRTXLiDAR)
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    sensor._debug_vis_handle = None
    sensor._renderer = Renderer()
    sensor._renderer_input_step = None
    sensor._view = None
    sensor._product_paths = ("/OVRTX/Products/Front", "/OVRTX/Products/Rear")
    sensor._num_envs = 2
    sensor._data = OVRTXLiDARData(num_envs=2)
    sensor._ALL_ENV_MASK = wp.ones(2, dtype=wp.bool, device="cpu")
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")

    sensor._update_buffers_impl(env_mask)

    assert events == [
        ("prepare", sensor._renderer, 12, 0.24),
        ("read", ("/OVRTX/Products/Front",)),
    ]
    torch.testing.assert_close(sensor._data.counts, torch.tensor([2, 0], dtype=torch.int32))
    torch.testing.assert_close(sensor._data.has_data, torch.tensor([True, False]))
    torch.testing.assert_close(sensor._data.is_fresh, torch.tensor([True, False]))

    sensor._update_buffers_impl(env_mask)

    torch.testing.assert_close(sensor._data.counts, torch.tensor([2, 0], dtype=torch.int32))
    torch.testing.assert_close(sensor._data.has_data, torch.tensor([True, False]))
    torch.testing.assert_close(sensor._data.is_fresh, torch.tensor([False, False]))
    sensor._renderer = None
    sensor._clear_callbacks()
