# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.cloner import ClonePlan
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXRenderData,
        OVRTXRenderer,
        ovrtx_use_ovstage_enabled,
    )
    from isaaclab_ov.sensors import (  # noqa: E402
        OVRTXLiDAR,
        OVRTXLiDARCfg,
        OVRTXLiDAROutputMetadata,
        OVRTXLiDARProductSpec,
    )
    from isaaclab_ov.sensors.lidar import ovrtx_lidar as ovrtx_lidar_module  # noqa: E402
else:
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    ovrtx_use_ovstage_enabled = None
    OVRTXLiDAR = None
    OVRTXLiDARCfg = None
    OVRTXLiDAROutputMetadata = None
    OVRTXLiDARProductSpec = None
    ovrtx_lidar_module = None

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.renderer_info = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._read_gpu_transforms = False
    renderer._render_product_paths = []
    renderer._camera_render_product_path = None
    renderer._lidar_product_specs = {}
    renderer._lidar_output_metadata = {}
    renderer._emitted_lidar_frames = {}
    renderer._last_step_products = None
    return renderer


def test_ovrtx_renderer_preserves_camera_gpu_transform_default(monkeypatch: pytest.MonkeyPatch):
    """Camera-only OVRTX keeps its established GPU-transform default."""
    renderer_configs = []
    monkeypatch.delenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", raising=False)
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "Renderer",
        lambda cfg: renderer_configs.append(cfg) or object(),
    )

    OVRTXRenderer(OVRTXRendererCfg())

    assert renderer_configs[0].motion_bvh is None
    assert renderer_configs[0].read_gpu_transforms is True


def test_ovrtx_renderer_cfg_can_disable_gpu_transforms_for_lidar(monkeypatch: pytest.MonkeyPatch):
    """LiDAR scenes disable the known-bad transform path through explicit renderer config."""
    renderer_configs = []
    monkeypatch.setenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "1")
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "Renderer",
        lambda cfg: renderer_configs.append(cfg) or object(),
    )

    OVRTXRenderer(OVRTXRendererCfg(read_gpu_transforms=False, motion_bvh="auto"))

    assert renderer_configs[0].read_gpu_transforms is False
    assert renderer_configs[0].motion_bvh == ovrtx_renderer_module.MotionBvh.AUTO


def test_ovrtx_frame_transactions_are_enabled_only_by_time_integrating_lidar():
    """Camera-only rendering stays on demand; LiDAR opts the shared renderer into simulation time."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False

    assert renderer.uses_frame_transactions() is False

    renderer.register_lidar_product(
        OVRTXLiDARProductSpec(
            sensor_prim_path="/World/envs/env_0/Lidar",
            product_path="/OVRTX/Products/Lidar_env_0",
        )
    )

    assert renderer.uses_frame_transactions() is True


@pytest.mark.parametrize("use_ovstage", (False, True))
def test_ovrtx_camera_only_render_remains_on_demand(use_ovstage: bool):
    """A camera-only renderer produces a frame without requiring a simulation-time transaction."""
    camera_path = "/OVRTX/Products/Camera"
    step_calls: list[dict] = []
    floor_calls: list[int] = []

    class Completion:
        def wait(self):
            return None

    class Stage:
        def advance_write_floor(self, *, ordinal):
            floor_calls.append(ordinal)
            return Completion()

    class Backend:
        def step(self, **kwargs):
            step_calls.append(kwargs)
            return {camera_path: SimpleNamespace(frames=[])}

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = use_ovstage
    renderer._initialized_scene = True
    renderer._renderer = Backend()
    renderer._stage = Stage()
    renderer._current_ordinal = 3
    renderer._render_product_paths = [camera_path]
    renderer._camera_render_product_path = camera_path

    renderer.render(_make_ovrtx_render_data())

    assert step_calls == [
        {
            "render_products": {camera_path},
            "delta_time": pytest.approx(1.0 / 60.0),
            **({"ordinal": 3} if use_ovstage else {}),
        }
    ]
    assert floor_calls == ([3] if use_ovstage else [])
    assert renderer._current_ordinal == (4 if use_ovstage else 3)


def test_ovrtx_camera_and_lidar_consume_one_shared_step(monkeypatch: pytest.MonkeyPatch):
    """Rendering a camera after the LiDAR transaction cannot advance the shared OVRTX clock twice."""
    camera_path = "/OVRTX/Products/Camera"
    lidar_path = "/OVRTX/Products/Lidar_env_0"
    step_calls: list[dict] = []

    class Completion:
        def wait(self):
            return None

    class Stage:
        def advance_write_floor(self, *, ordinal):
            assert ordinal == 0
            return Completion()

    class Backend:
        def step(self, **kwargs):
            step_calls.append(kwargs)
            return {
                camera_path: SimpleNamespace(frames=[]),
                lidar_path: SimpleNamespace(frames=[]),
            }

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = True
    renderer._renderer = Backend()
    renderer._stage = Stage()
    renderer._current_ordinal = 0
    renderer._render_product_paths = [camera_path, lidar_path]
    renderer._camera_render_product_path = camera_path
    renderer._device = "cuda:0"
    renderer._lidar_product_specs = {
        lidar_path: OVRTXLiDARProductSpec(
            sensor_prim_path="/World/envs/env_0/Lidar",
            product_path=lidar_path,
        )
    }
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0))

    renderer.advance_frame(0.02)
    renderer.render(_make_ovrtx_render_data())

    assert step_calls == [
        {
            "render_products": {camera_path, lidar_path},
            "delta_time": 0.02,
            "ordinal": 0,
        }
    ]


def test_ovrtx_lidar_rejects_explicit_gpu_transform_propagation(monkeypatch: pytest.MonkeyPatch):
    """LiDAR fails fast for OVRTX's known-bad GPU-transform/geometry-streaming combination."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "1")
    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda cfg: object())
    renderer = OVRTXRenderer(OVRTXRendererCfg())
    spec = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/Lidar",
        product_path="/OVRTX/Products/Lidar",
    )

    with pytest.raises(RuntimeError, match="read_gpu_transforms=True.*incompatible.*LiDAR"):
        renderer.register_lidar_product(spec)


def test_ovrtx_lidar_cfg_rejects_non_ovrtx_renderer():
    """An OVRTX LiDAR cannot use a renderer that cannot produce its point cloud."""
    cfg = OVRTXLiDARCfg(prim_path="/World/Lidar", renderer_cfg=object())

    with pytest.raises(TypeError, match="OVRTXRendererCfg"):
        cfg.validate()


def test_ovrtx_lidar_cfg_disables_gpu_transforms_by_default():
    """The opt-in LiDAR config is safe without changing Camera-only renderer defaults."""
    cfg = OVRTXLiDARCfg(prim_path="/World/Lidar")

    assert cfg.renderer_cfg.read_gpu_transforms is False
    assert cfg.renderer_cfg.motion_bvh == "auto"


@pytest.mark.parametrize(
    ("renderer_cfg", "message"),
    [
        (OVRTXRendererCfg(), "read_gpu_transforms=False"),
        (OVRTXRendererCfg(read_gpu_transforms=True, motion_bvh="auto"), "read_gpu_transforms=False"),
        (OVRTXRendererCfg(read_gpu_transforms=False), "motion_bvh"),
        (OVRTXRendererCfg(read_gpu_transforms=False, motion_bvh="disable"), "motion_bvh"),
    ],
)
def test_ovrtx_lidar_cfg_rejects_implicit_or_incompatible_renderer_paths(renderer_cfg, message):
    """LiDAR requires explicit transform and motion settings instead of environment-dependent behavior."""
    cfg = OVRTXLiDARCfg(prim_path="/World/Lidar", renderer_cfg=renderer_cfg)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_ovrtx_renderer_rejects_unknown_motion_bvh_mode(monkeypatch: pytest.MonkeyPatch):
    """Invalid backend enum values fail before OVRTX renderer construction."""
    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda _cfg: object())

    with pytest.raises(ValueError, match="motion_bvh"):
        OVRTXRenderer(OVRTXRendererCfg(motion_bvh="sometimes"))


def test_ovrtx_lidar_cfg_requests_fidelity_metadata_by_default():
    """The default layout preserves beam and timing identity, not only XYZ."""
    cfg = OVRTXLiDARCfg(prim_path="/World/Lidar")

    assert cfg.channels == (
        "Coordinates",
        "Intensity",
        "TimeOffsetNs",
        "EmitterId",
        "ChannelId",
        "TickId",
        "EchoId",
    )


def test_ovrtx_transaction_reset_uses_authoritative_time_and_drops_sensor_history():
    """Reset must not expose pre-reset outputs or advance OVRTX with fabricated time."""
    reset_times: list[float] = []
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._renderer = SimpleNamespace(reset=lambda simulation_time: reset_times.append(simulation_time))
    renderer._last_step_products = object()
    renderer._emitted_lidar_frames = {"/OVRTX/Products/Lidar": object()}

    renderer.reset_frame_transaction(1.25)

    assert reset_times == [1.25]
    assert renderer._last_step_products is None
    assert renderer._emitted_lidar_frames == {}


def test_ovrtx_lidar_prepares_one_product_per_clone(monkeypatch: pytest.MonkeyPatch):
    """The sensor expands the clone plan before the renderer root layer is frozen."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.tensor([0, 1]),
    )
    monkeypatch.setattr(
        ovrtx_lidar_module,
        "SimulationContext",
        SimpleNamespace(instance=lambda: SimpleNamespace(get_clone_plan=lambda: plan)),
    )
    sensor = OVRTXLiDAR.__new__(OVRTXLiDAR)
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    sensor._debug_vis_handle = None
    sensor.cfg = OVRTXLiDARCfg(prim_path="/World/envs/env_.*/Robot/Lidar")
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {}

    sensor.prepare_renderer(renderer)

    specs = tuple(renderer._lidar_product_specs.values())
    assert tuple(spec.sensor_prim_path for spec in specs) == (
        "/World/envs/env_0/Robot/Lidar",
        "/World/envs/env_1/Robot/Lidar",
    )
    assert len({spec.product_path for spec in specs}) == 2
    assert sensor.product_paths == tuple(spec.product_path for spec in specs)


def test_ovrtx_lidar_rejects_multiple_matches_in_one_environment(monkeypatch: pytest.MonkeyPatch):
    """One sensor cfg cannot silently choose between multiple LiDAR prims in an environment."""
    plan = SimpleNamespace(
        env_ids=torch.tensor([0]),
        clone_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    monkeypatch.setattr(
        ovrtx_lidar_module,
        "SimulationContext",
        SimpleNamespace(instance=lambda: SimpleNamespace(get_clone_plan=lambda: plan)),
    )
    monkeypatch.setattr(
        ovrtx_lidar_module.cloner,
        "iter_clone_plan_matches",
        lambda _plan, _expression: iter(
            (
                ("/World/envs/env_0", "/World/envs/env_{}", "/World/envs/env_0/FrontLidar", (0,)),
                ("/World/envs/env_0", "/World/envs/env_{}", "/World/envs/env_0/RearLidar", (0,)),
            )
        ),
    )
    sensor = OVRTXLiDAR.__new__(OVRTXLiDAR)
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    sensor._debug_vis_handle = None
    sensor.cfg = OVRTXLiDARCfg(prim_path="/World/envs/env_.*/.*Lidar")
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {}

    with pytest.raises(RuntimeError, match="multiple OVRTX LiDAR prims.*environment 0"):
        sensor.prepare_renderer(renderer)


def test_ovrtx_lidar_product_requires_exactly_one_sensor_prim():
    """Each point-cloud product has one LiDAR source because OVRTX point clouds are not tiled."""
    with pytest.raises(ValueError, match="exactly one"):
        OVRTXLiDARProductSpec(sensor_prim_path="", product_path="/OVRTX/Products/Lidar")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_prim_path": "World/Lidar"}, "absolute"),
        ({"product_path": "/Render/Products/Lidar"}, "/OVRTX/Products"),
        ({"channels": ("Intensity",)}, "Coordinates"),
        ({"channels": ("Coordinates", "Coordinates")}, "duplicate"),
    ],
)
def test_ovrtx_lidar_product_rejects_ambiguous_layouts(kwargs, message):
    """Invalid paths and ambiguous channel layouts fail before USD compilation."""
    values = {
        "sensor_prim_path": "/World/Lidar",
        "product_path": "/OVRTX/Products/Lidar",
        "channels": ("Coordinates", "Intensity"),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        OVRTXLiDARProductSpec(**values)


def test_ovrtx_lidar_product_can_register_before_scene_load():
    """The shared renderer accepts LiDAR products while its root layer is still mutable."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {}
    spec = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/Lidar",
        product_path="/OVRTX/Products/Lidar_env_0",
    )

    renderer.register_lidar_product(spec)


def test_ovrtx_lidar_product_rejects_conflicting_product_registration():
    """A product path cannot silently switch to a different LiDAR source."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {}
    first = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/FrontLidar",
        product_path="/OVRTX/Products/Lidar_env_0",
    )
    conflicting = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/RearLidar",
        product_path=first.product_path,
    )
    renderer.register_lidar_product(first)

    with pytest.raises(ValueError, match="already registered"):
        renderer.register_lidar_product(conflicting)


def test_ovrtx_lidar_product_rejects_registration_after_scene_load():
    """ovstage has one root layer, so a product cannot be added after scene load."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = True
    renderer._lidar_product_specs = {}
    spec = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/Lidar",
        product_path="/OVRTX/Products/Lidar_env_0",
    )

    with pytest.raises(RuntimeError, match="before.*scene"):
        renderer.register_lidar_product(spec)


def test_ovrtx_lidar_product_rejects_legacy_renderer_path():
    """LiDAR requires ovstage and never falls back to OVRTX's deprecated scene ownership path."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = False
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {}
    spec = OVRTXLiDARProductSpec(
        sensor_prim_path="/World/envs/env_0/Lidar",
        product_path="/OVRTX/Products/Lidar_env_0",
    )

    with pytest.raises(RuntimeError, match="ovstage"):
        renderer.register_lidar_product(spec)


def test_ovrtx_lidar_scene_initialization_is_explicit_and_idempotent(monkeypatch: pytest.MonkeyPatch):
    """LiDAR-only scenes load lazily after every sensor has registered its product."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = False
    renderer._lidar_product_specs = {
        "/OVRTX/Products/Lidar": OVRTXLiDARProductSpec(
            sensor_prim_path="/World/envs/env_0/Lidar",
            product_path="/OVRTX/Products/Lidar",
        )
    }
    calls = []

    def initialize(num_envs: int):
        calls.append(num_envs)
        renderer._initialized_scene = True

    monkeypatch.setattr(renderer, "_initialize_lidar_scene_ovstage", initialize)

    renderer.initialize_lidar_scene(1)
    renderer.initialize_lidar_scene(1)

    assert calls == [1]


def test_ovrtx_lidar_read_reports_no_frame_as_normal_scan_cadence():
    """A transaction without PointCloud output returns no sample instead of raising or fabricating one."""
    product_path = "/OVRTX/Products/Lidar"
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._lidar_product_specs = {
        product_path: OVRTXLiDARProductSpec(
            sensor_prim_path="/World/envs/env_0/Lidar",
            product_path=product_path,
        )
    }

    assert renderer.read_lidar((product_path,)) == {}


def test_ovrtx_lidar_scene_open_rejects_multi_environment_scene_partitions():
    """OVRTX 0.4 must not silently return empty LiDAR frames for partitioned cloned geometry."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._exported_usd_string = '#usda 1.0\ndef Xform "World" {}'
    renderer._lidar_product_specs = {
        "/OVRTX/Products/Lidar": OVRTXLiDARProductSpec(
            sensor_prim_path="/World/envs/env_0/Lidar",
            product_path="/OVRTX/Products/Lidar",
        )
    }

    with pytest.raises(RuntimeError, match=r"OVRTX 0\.4.*scene-partitioned geometry.*2 environments"):
        renderer._open_scene_ovstage(2, ())

    assert renderer._exported_usd_string == '#usda 1.0\ndef Xform "World" {}'


@pytest.mark.parametrize(("torch_stream", "ovrtx_stream"), ((0, 1), (123, 123)))
def test_ovrtx_frame_transaction_maps_owned_cuda_lidar_frames(
    monkeypatch: pytest.MonkeyPatch,
    torch_stream: int,
    ovrtx_stream: int,
):
    """One global step caches owned LiDAR frames and leaves no active mappings."""

    class Completion:
        def wait(self):
            return

    class Stage:
        def advance_write_floor(self, *, ordinal):
            assert ordinal == 4
            return Completion()

    class Mapping(dict):
        def __init__(self, tensors):
            super().__init__(tensors)
            self.params = {"frameId": torch.tensor(9, dtype=torch.uint64)}
            self.unmapped_stream = None

        def unmap(self, *, stream):
            self.unmapped_stream = stream

    class RenderVar:
        def __init__(self, mapping):
            self.mapping = mapping

        def map(self, *, device, sync_stream):
            assert device == ovrtx_renderer_module.Device.CUDA
            assert sync_stream == ovrtx_stream
            return self.mapping

    source_coordinates = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mappings = {
        path: Mapping(
            {
                "Coordinates": source_coordinates + index,
                "Counts": torch.tensor([4], dtype=torch.int32),
                "Flags": torch.full((4,), 0x40, dtype=torch.uint8),
            }
        )
        for index, path in enumerate(("/OVRTX/Products/Front", "/OVRTX/Products/Rear"))
    }

    class Backend:
        def step(self, *, render_products, delta_time, ordinal):
            assert render_products == {"/OVRTX/Products/Camera", *mappings}
            assert delta_time == 0.02
            assert ordinal == 4
            return {
                path: SimpleNamespace(frames=[SimpleNamespace(render_vars={"PointCloud": RenderVar(mapping)})])
                for path, mapping in mappings.items()
            }

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=torch_stream))
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._initialized_scene = True
    renderer._stage = Stage()
    renderer._renderer = Backend()
    renderer._current_ordinal = 4
    renderer._camera_render_product_path = "/OVRTX/Products/Camera"
    renderer._render_product_paths = [renderer._camera_render_product_path, *mappings]
    renderer._lidar_product_specs = {
        path: OVRTXLiDARProductSpec(
            sensor_prim_path=f"/World/envs/env_{index}/Lidar",
            product_path=path,
        )
        for index, path in enumerate(mappings)
    }
    renderer._lidar_output_metadata = {
        path: OVRTXLiDAROutputMetadata(
            coordinate_frame="SENSOR",
            motion_compensated=False,
            partial_outputs=False,
            instant_lidar=True,
            frame_rate_hz=10.0,
            max_returns=1,
        )
        for path in mappings
    }
    renderer._emitted_lidar_frames = {}
    renderer._last_step_products = None
    renderer._device = "cuda:0"

    renderer.advance_frame(0.02)
    frames = renderer.read_lidar(tuple(mappings))

    assert set(frames) == set(mappings)
    assert renderer._current_ordinal == 5
    assert frames["/OVRTX/Products/Front"].tensors["Coordinates"].data_ptr() != source_coordinates.data_ptr()
    assert all(mapping.unmapped_stream == ovrtx_stream for mapping in mappings.values())


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
        RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.NORMALS,
        RenderBufferKind.MOTION_VECTORS,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, wp.uint8)
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, wp.float32)
    assert specs[RenderBufferKind.MOTION_VECTORS] == RenderBufferSpec(2, wp.float32)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert render_data.warp_buffers["depth"].ptr == data.output["depth"].warp.ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_set_outputs_wraps_requested_rgb_hdr_output():
    """OVRTXRenderer.set_outputs publishes a zero-copy view for requested RGB_HDR."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb_hdr"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgb_hdr"].ptr == data.output["rgb_hdr"].warp.ptr


def test_ovrtx_set_outputs_routes_ppisp_buffers_through_warp_buffers():
    """OVRTXRenderer.set_outputs stores PPISP source/destination in warp_buffers."""
    renderer = _make_ovrtx_renderer_without_backend()

    cfg = _make_camera_cfg(["rgb"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert "rgb_hdr" in render_data.warp_buffers
    assert render_data.warp_buffers["rgb_hdr"].shape == (2, 8, 16, 3)
    assert render_data.warp_buffers["rgb_hdr"].dtype is wp.float32


def test_ovrtx_process_frame_skips_ldr_rgba_when_ppisp_is_active():
    """PPISP owns RGBA output, so OVRTX LdrColor should not pre-fill it."""

    class FailingRenderVar:
        def map(self, *args, **kwargs):
            raise AssertionError("PPISP RGBA output must not read OVRTX LdrColor")

    class Frame:
        render_vars = {"LdrColor": FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


def test_ovrtx_ppisp_hdr_source_is_cloned_to_output_device(monkeypatch):
    """PPISP HdrColor source is moved to the HDR output buffer device."""

    class FakeArray:
        device = "cuda:1"

    class OutputArray:
        device = "cuda:0"

    cloned = object()
    clone_calls = []

    def fake_clone(src, *, device):
        clone_calls.append((src, device))
        return cloned

    monkeypatch.setattr(wp, "clone", fake_clone)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    source = FakeArray()

    assert renderer._prepare_ppisp_hdr_source(render_data, source, {"rgb_hdr": OutputArray()}) is cloned
    assert clone_calls == [(source, "cuda:0")]


class _FakeArray:
    def __init__(self, shape):
        self.shape = shape


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, _FakeArray((8, 16, 3)), _FakeArray((2, 8, 16, 4)))


def test_launch_extract_all_tiles_launches_kernel_when_channels_are_compatible(monkeypatch):
    """Equal or narrower output channel counts pass validation and reach the kernel launch."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()
    render_data.num_cols = 2

    launch_calls = []
    monkeypatch.setattr(wp, "launch", lambda **kwargs: launch_calls.append(kwargs))

    tiled_buffer = _FakeArray((8, 16, 4))
    output_buffer = _FakeArray((2, 8, 16, 3))
    renderer._launch_extract_all_tiles(render_data, tiled_buffer, output_buffer)

    assert len(launch_calls) == 1
    assert launch_calls[0]["inputs"][:2] == [tiled_buffer, output_buffer]


def test_ovrtx_read_output_copies_no_pixel_data():
    """OVRTXRenderer.read_output copies no pixel data; with empty renderer_info it leaves info untouched."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}


def test_ovrtx_read_output_forwards_renderer_info():
    """OVRTXRenderer.read_output forwards render_data.renderer_info (e.g. semantic idToLabels) into info."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}

    camera_data = CameraData()
    camera_data.info = {"semantic_segmentation": None}
    camera_data._output = {}

    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}


def test_ovrtx_read_output_clears_stale_metadata_and_keeps_seeded_keys():
    """read_output replaces (not merges): a dropped render var resets its info entry, seeded keys persist."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}
    camera_data._output = {}

    # Frame 1: the SemanticIdMap render var is present, so its metadata lands in info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: render() rebuilds renderer_info from scratch and the SemanticIdMap is gone this frame.
    render_data.renderer_info = {}
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}


def test_ovrtx_semantic_spec_follows_colorize_flag():
    """Semantic segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_instance_segmentation_spec_follows_colorize_flag():
    """Instance segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_use_ovstage_defaults_to_disabled(monkeypatch):
    """The ovstage path is off unless explicitly opted into, so existing deployments are unaffected."""
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    assert ovrtx_use_ovstage_enabled() is False

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0")
    assert ovrtx_use_ovstage_enabled() is False


def test_ovrtx_use_ovstage_enabled_when_requested_and_available(monkeypatch):
    """Setting the variable to 1 selects the ovstage path when ovstage is importable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", True)
    assert ovrtx_use_ovstage_enabled() is True


def test_ovrtx_use_ovstage_raises_when_requested_but_unavailable(monkeypatch):
    """An explicit opt-in must fail loudly rather than silently falling back to the legacy path."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="ov\\[ovstage\\]"):
        ovrtx_use_ovstage_enabled()


def test_ovrtx_use_ovstage_rejects_non_boolean_values(monkeypatch):
    """Values other than 0/1 are a configuration error, not a silent disable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "true")
    monkeypatch.setattr(ovrtx_renderer_module, "_OVSTAGE_AVAILABLE", True)

    with pytest.raises(ValueError, match="Expected 0 or 1"):
        ovrtx_use_ovstage_enabled()


def test_ovrtx_cleanup_releases_only_the_given_render_data():
    """``cleanup`` releases the render data's own buffers and leaves the renderer usable.

    The stage queries, tensor bindings and render products the renderer holds are shared with
    every other camera that resolved to it, so a single camera's cleanup must not take them.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    render_data = _make_ovrtx_render_data()
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()

    renderer.cleanup(render_data)

    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.ppisp_pipeline is None

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


def test_ovrtx_cleanup_without_render_data_keeps_renderer_state():
    """``cleanup(None)`` has nothing to release and must not disturb the renderer."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    renderer.cleanup(None)

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


class _RecordingBinding:
    def __init__(self, events: list[str], name: str):
        self._events = events
        self._name = name

    def unbind(self) -> None:
        self._events.append(f"unbind:{self._name}")


def _make_legacy_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build a legacy-path renderer whose backend calls are recorded into ``events``."""

    class Backend:
        def reset_stage(self) -> None:
            events.append("reset_stage")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = False
    renderer._camera_xform_binding = _RecordingBinding(events, "camera")
    renderer._object_xform_binding = _RecordingBinding(events, "object")
    renderer._deformable_points_binding = _RecordingBinding(events, "deformable")
    renderer._particle_points_binding = _RecordingBinding(events, "particle")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._particle_workaround_applied = True
    renderer._renderer = Backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._camera_render_product_path = "/Render/RenderProduct_camera"
    renderer._lidar_product_specs = {}
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    return renderer


def _make_ovstage_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build an ovstage-path renderer whose backend calls are recorded into ``events``."""

    class Completion:
        def wait(self) -> None:
            return

    class Stage:
        def release_query(self, query):
            events.append(f"release_query:{query}")
            return Completion()

    class StagePaths:
        def destroy_path_list(self, path_list) -> None:
            events.append(f"destroy_path_list:{path_list}")

    class Backend:
        def detach_ovstage(self) -> None:
            events.append("detach_ovstage")

    class ExitStack:
        def close(self) -> None:
            events.append("exit_stack_close")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._stage = Stage()
    renderer._stage_paths = StagePaths()
    renderer._camera_xform_query = "camera"
    renderer._camera_paths_list = "camera"
    renderer._object_xform_query = "object"
    renderer._object_paths_list = "object"
    renderer._deformable_points_query = "deformable"
    renderer._deformable_paths_list = "deformable"
    renderer._particle_points_query = "particle"
    renderer._particle_paths_list = "particle"
    renderer._object_newton_indices = object()
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._env_root_xforms = object()
    renderer._renderer = Backend()
    renderer._ovstage_exit_stack = ExitStack()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._camera_render_product_path = "/Render/RenderProduct_camera"
    renderer._lidar_product_specs = {}
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    renderer._current_ordinal = 7
    return renderer


def test_ovrtx_close_releases_legacy_renderer_state():
    """``close`` unbinds the tensor bindings and resets the stage the renderer owns."""
    events: list[str] = []
    renderer = _make_legacy_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "unbind:camera",
        "unbind:object",
        "unbind:deformable",
        "unbind:particle",
        "reset_stage",
    ]
    assert renderer._camera_xform_binding is None
    assert renderer._object_xform_binding is None
    assert renderer._deformable_points_binding is None
    assert renderer._particle_points_binding is None
    assert renderer._particle_workaround_applied is False
    assert renderer._renderer is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False


def test_ovrtx_close_releases_ovstage_renderer_state():
    """``close`` releases the queries and path lists, then detaches before closing the ExitStack.

    The ExitStack owns the ovstage ``Stage`` and ``PathDictionary`` as context managers, so it is the
    only thing that releases them — ``ExitStack`` has no finalizer, and garbage collection never
    invokes ``__exit__``. Detaching first avoids a use-after-free while the renderer still references
    the stage.
    """
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "release_query:camera",
        "destroy_path_list:camera",
        "release_query:object",
        "destroy_path_list:object",
        "release_query:deformable",
        "destroy_path_list:deformable",
        "release_query:particle",
        "destroy_path_list:particle",
        "detach_ovstage",
        "exit_stack_close",
    ]
    assert renderer._camera_xform_query is None
    assert renderer._particle_paths_list is None
    assert renderer._object_newton_indices is None
    assert renderer._env_root_xforms is None
    assert renderer._renderer is None
    assert renderer._ovstage_exit_stack is None
    assert renderer._stage is None
    assert renderer._stage_paths is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False
    assert renderer._current_ordinal == 0


def test_ovrtx_close_is_idempotent():
    """A second ``close`` releases nothing again, so a repeated teardown cannot double-free."""
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()
    events.clear()
    renderer.close()

    assert events == []
