# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU integration tests for the public OVRTX LiDAR renderer seam."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "isaaclab_newton", "ovrtx", "ovstage", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.kitless,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="OVRTX LiDAR requires a CUDA device"),
]

if not _MISSING_MODULES:
    import warp as wp  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRenderer, OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.sensors import OVRTXLiDARProductSpec  # noqa: E402

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

    from isaaclab.cloner import ClonePlan  # noqa: E402
    from isaaclab.renderers import CameraRenderSpec  # noqa: E402
    from isaaclab.sensors.camera import CameraCfg, CameraData  # noqa: E402
    from isaaclab.sim import PinholeCameraCfg  # noqa: E402
    from isaaclab.utils.warp import ProxyArray  # noqa: E402
else:
    CameraCfg = None
    CameraData = None
    CameraRenderSpec = None
    ClonePlan = None
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    OVRTXLiDARProductSpec = None
    PinholeCameraCfg = None
    ovrtx_renderer_module = None
    Usd = None
    wp = None


_SCENE = r"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "envs"
    {
        def Xform "env_0"
        {
            def OmniLidar "Lidar" (
                prepend apiSchemas = ["OmniSensorGenericLidarCoreAPI"]
            )
            {
                token omni:sensor:Core:elementsCoordsType = "CARTESIAN"
                token omni:sensor:Core:outputFrameOfReference = "SENSOR"
                token omni:sensor:Core:outputMotionCompensationState = "NONCOMPENSATED"
                float omni:sensor:Core:azimuthErrorMean = 0
                float omni:sensor:Core:azimuthErrorStd = 0
                float omni:sensor:Core:elevationErrorMean = 0
                float omni:sensor:Core:elevationErrorStd = 0
                float[] omni:sensor:Core:originErrorMean = [0, 0, 0]
                float[] omni:sensor:Core:originErrorStd = [0, 0, 0]
                float omni:sensor:Core:rangeAccuracyM = 0
                float omni:sensor:Core:rangeResolutionM = 0.001
                float omni:sensor:Core:nearRangeM = 0.1
                float omni:sensor:Core:farRangeM = 20
                bool omni:sensor:Core:instantLidar = 1
                bool omni:sensor:Core:partialOutputs = 0
                uint omni:sensor:Core:maxReturns = 1
                double2 omni:sensor:frameRate = (10, 1)
                double3 xformOp:translate = (1, 2, 1)
                float3 xformOp:rotateXYZ = (90, 0, -90)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
            }

            def Camera "Camera"
            {
                float2 clippingRange = (0.1, 20)
                float focalLength = 24
                float focusDistance = 400
                float horizontalAperture = 20.955
                float verticalAperture = 15.2908
                double3 xformOp:translate = (1, 2, 1)
                float3 xformOp:rotateXYZ = (90, 0, -90)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
            }

            def Cube "WallPositiveX" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)]
                double size = 2
                rel material:binding = </World/TestSurface>
                double3 xformOp:scale = (0.05, 8, 8)
                double3 xformOp:translate = (5, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }

            def Cube "OccluderPositiveX" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                color3f[] primvars:displayColor = [(0.1, 0.8, 0.1)]
                double size = 2
                rel material:binding = </World/TestSurface>
                double3 xformOp:scale = (0.25, 0.5, 0.5)
                double3 xformOp:translate = (3, 2, 1)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }

            def Cube "WallNegativeX" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                double size = 2
                rel material:binding = </World/TestSurface>
                double3 xformOp:scale = (0.05, 8, 8)
                double3 xformOp:translate = (-6, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }

            def Cube "WallPositiveY" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                double size = 2
                rel material:binding = </World/TestSurface>
                double3 xformOp:scale = (8, 0.05, 8)
                double3 xformOp:translate = (0, 7, 0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }

            def Cube "WallNegativeY" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                double size = 2
                rel material:binding = </World/TestSurface>
                double3 xformOp:scale = (8, 0.05, 8)
                double3 xformOp:translate = (0, -8, 0)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
        }
    }

    def Material "TestSurface"
    {
        token omni:simready:nonvisual:base = "concrete"
        token omni:simready:nonvisual:coating = "none"
        token[] omni:simready:nonvisual:attributes = ["none"]
        custom string inputs:nonvisual:base = "concrete"
        custom string inputs:nonvisual:coating = "none"
        custom string inputs:nonvisual:attributes = "none"
    }
}
"""


def _make_renderer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body_paths: tuple[str, ...] = (),
    body_q: object | None = None,
) -> tuple[Usd.Stage, OVRTXRenderer]:
    """Create one clean-room stage and its sole OVRTX renderer."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    stage = Usd.Stage.CreateInMemory()
    assert stage.GetRootLayer().ImportFromString(_SCENE)
    clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 1), dtype=torch.bool),
        env_ids=torch.tensor([0]),
    )
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "SimulationContext",
        SimpleNamespace(instance=lambda: SimpleNamespace(get_clone_plan=lambda: clone_plan)),
    )
    if body_paths:
        if body_q is None:
            raise ValueError("body_q is required when Newton body paths are provided")
        monkeypatch.setattr(
            NewtonManager,
            "get_model",
            classmethod(lambda cls: SimpleNamespace(body_label=body_paths)),
        )
        monkeypatch.setattr(
            NewtonManager,
            "get_state",
            classmethod(lambda cls: SimpleNamespace(body_q=body_q)),
        )
    else:
        monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: None))
    return stage, OVRTXRenderer(OVRTXRendererCfg(log_level="warn", read_gpu_transforms=False, motion_bvh="auto"))


def _lidar_product_spec(
    *,
    sensor_prim_path: str = "/World/envs/env_0/Lidar",
    product_path: str = "/OVRTX/Products/DeterministicLidar",
) -> OVRTXLiDARProductSpec:
    """Return the deterministic PointCloud product shared by GPU tests."""
    return OVRTXLiDARProductSpec(
        sensor_prim_path=sensor_prim_path,
        product_path=product_path,
        channels=(
            "Coordinates",
            "Intensity",
            "TimeOffsetNs",
            "EmitterId",
            "ChannelId",
            "TickId",
            "EchoId",
        ),
    )


def _valid_points(frame) -> torch.Tensor:
    """Return only model-declared valid Cartesian points from one frame."""
    count = int(frame.tensors["Counts"][0])
    assert count > 0, {
        "frameId": int(frame.params["frameId"].reshape(-1)[0]),
        "maxPoints": int(frame.params["maxPoints"].reshape(-1)[0]),
        "timestampNs": int(frame.params["timestampNs"].reshape(-1)[0]),
    }
    valid = (frame.tensors["Flags"][:count] & 0x40) != 0
    assert bool(valid.any()), {"count": count, "flags": torch.unique(frame.tensors["Flags"][:count]).tolist()}
    return frame.tensors["Coordinates"][:, :count].T[valid]


def _warm_up_lidar(renderer: OVRTXRenderer, spec: OVRTXLiDARProductSpec, *, steps: int = 3) -> None:
    """Run the warm-up frames required by the upstream OVRTX LiDAR example."""
    for _ in range(steps):
        renderer.advance_frame(0.1)
        renderer.read_lidar((spec.product_path,))


def _advance_and_read_lidar(
    renderer: OVRTXRenderer,
    product_paths: tuple[str, ...],
    *,
    delta_time: float = 0.1,
):
    """Advance the shared renderer once, then read requested LiDAR products without another step."""
    renderer.advance_frame(delta_time)
    return renderer.read_lidar(product_paths)


def _read_bound_object_xforms(renderer: OVRTXRenderer, ordinal: int) -> np.ndarray:
    """Read the Newton-bound ``omni:xform`` matrices through the official ovstage API."""
    assert renderer._stage is not None
    assert renderer._stage_paths is not None
    assert renderer._object_xform_query is not None
    attribute = renderer._stage_paths.intern_token("omni:xform")
    ordinal_range = ovrtx_renderer_module.ovstage.OrdinalRange.latest(ordinal)
    with renderer._stage.read_attributes(renderer._object_xform_query, [attribute], ordinal_range) as read:
        group = read.fetch_next()
        assert group is not None
        try:
            return np.from_dlpack(group.dlpack(0)).copy().reshape(-1, 4, 4)
        finally:
            renderer._stage.release_group(group)


def _replace_static_occluder_with_dynamic_target(stage: Usd.Stage, target_path: str) -> None:
    """Replace the fixture occluder with geometry below a Newton-controlled parent xform."""
    assert stage.RemovePrim("/World/envs/env_0/OccluderPositiveX")
    target = UsdGeom.Xform.Define(stage, target_path)
    UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
    target.AddTranslateOp().Set(Gf.Vec3d(3.0, 2.0, 1.0))
    geometry = UsdGeom.Cube.Define(stage, f"{target_path}/Geometry")
    geometry.CreateSizeAttr(2.0)
    UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    UsdGeom.Xformable(geometry).AddScaleOp().Set(Gf.Vec3d(0.25, 0.5, 0.5))
    material = UsdShade.Material(stage.GetPrimAtPath("/World/TestSurface"))
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def _move_lidar_below_mount(stage: Usd.Stage, mount_path: str) -> str:
    """Move the authored LiDAR below a Newton-controlled parent without changing its local rotation."""
    sensor_path = f"{mount_path}/Lidar"
    mount = UsdGeom.Xform.Define(stage, mount_path)
    UsdPhysics.RigidBodyAPI.Apply(mount.GetPrim())
    mount.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 1.0))
    layer = stage.GetRootLayer()
    assert Sdf.CopySpec(layer, Sdf.Path("/World/envs/env_0/Lidar"), layer, Sdf.Path(sensor_path))
    assert stage.RemovePrim("/World/envs/env_0/Lidar")
    assert stage.GetPrimAtPath(sensor_path).GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return sensor_path


def test_public_renderer_seam_produces_deterministic_sensor_frame_pointcloud(monkeypatch: pytest.MonkeyPatch):
    """Distance, occlusion, coordinates, channels, dtypes, and CUDA placement hold end to end."""
    stage, renderer = _make_renderer(monkeypatch)
    spec = _lidar_product_spec()
    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]
        torch.cuda.synchronize()

        assert frame.metadata.coordinate_frame == "SENSOR"
        assert frame.metadata.motion_compensated is False
        assert frame.metadata.partial_outputs is False
        assert frame.metadata.instant_lidar is True
        assert frame.metadata.frame_rate_hz == pytest.approx(10.0)
        assert frame.metadata.max_returns == 1
        assert frame.metadata.scan_complete is True

        expected_layout = {
            "Coordinates": (torch.float32, 2),
            "Intensity": (torch.float32, 1),
            "TimeOffsetNs": (torch.int32, 1),
            "EmitterId": (torch.uint32, 1),
            "ChannelId": (torch.uint32, 1),
            "TickId": (torch.uint32, 1),
            "EchoId": (torch.uint8, 1),
            "Counts": (torch.int32, 1),
            "Flags": (torch.uint8, 1),
        }
        assert set(frame.tensors) == set(expected_layout)
        for name, (dtype, ndim) in expected_layout.items():
            tensor = frame.tensors[name]
            assert tensor.dtype == dtype
            assert tensor.ndim == ndim
            assert tensor.is_cuda

        coordinates = frame.tensors["Coordinates"]
        assert coordinates.shape[0] == 3
        capacity = coordinates.shape[1]
        for name, tensor in frame.tensors.items():
            if name not in {"Coordinates", "Counts"}:
                assert tensor.shape == (capacity,)

        count = int(frame.tensors["Counts"][0])
        assert 0 < count <= capacity
        valid = (frame.tensors["Flags"][:count] & 0x40) != 0
        assert bool(valid.all())
        points = coordinates[:, :count].T[valid]

        assert float(points[:, 0].max()) == pytest.approx(3.95, abs=0.03)
        assert float(points[:, 0].min()) == pytest.approx(-6.95, abs=0.03)
        assert float(points[:, 1].max()) == pytest.approx(4.95, abs=0.03)
        assert float(points[:, 1].min()) == pytest.approx(-9.95, abs=0.03)

        occluder_window = points[(points[:, 0] > 0.0) & (points[:, 1].abs() < 0.35) & (points[:, 2].abs() < 0.35)]
        assert len(occluder_window) > 1000
        assert float(occluder_window[:, 0].median()) == pytest.approx(1.75, abs=0.005)
        assert float(occluder_window[:, 0].max()) < 1.77

        assert int(frame.params["coordsType"].reshape(-1)[0]) == 1
        assert int(frame.params["frameOfReference"].reshape(-1)[0]) == 0
        assert int(frame.params["motionCompensationState"].reshape(-1)[0]) == 0
        assert int(frame.params["maxPoints"].reshape(-1)[0]) == capacity
        for name in ("frameStartPosM", "frameEndPosM"):
            position = frame.params[name]
            torch.testing.assert_close(
                position,
                position.new_tensor((1.0, 2.0, 1.0)),
                rtol=0.0,
                atol=1.0e-6,
            )
        model_to_app = frame.params["modelToAppTransform"]
        torch.testing.assert_close(
            model_to_app,
            torch.eye(4, dtype=model_to_app.dtype, device=model_to_app.device),
            rtol=0.0,
            atol=1.0e-6,
        )
    finally:
        renderer.close()


def test_dynamic_lidar_pose_preserves_official_360_degree_model_axes(
    monkeypatch: pytest.MonkeyPatch,
):
    """An IsaacLab world pose retains the fixed OmniLidar model-axis rotation."""
    stage, renderer = _make_renderer(monkeypatch)
    spec = _lidar_product_spec()
    positions = ProxyArray(wp.array([(1.0, 2.0, 1.0)], dtype=wp.vec3f, device="cuda:0"))
    orientations = ProxyArray(wp.array([(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4f, device="cuda:0"))

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        renderer.update_lidar((spec.product_path,), positions, orientations)
        frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]
        torch.cuda.synchronize()

        points = _valid_points(frame)
        assert float(points[:, 0].max()) == pytest.approx(3.95, abs=0.03)
        assert float(points[:, 0].min()) == pytest.approx(-6.95, abs=0.03)
        assert float(points[:, 1].max()) == pytest.approx(4.95, abs=0.03)
        assert float(points[:, 1].min()) == pytest.approx(-9.95, abs=0.03)
    finally:
        renderer.close()


def test_multi_return_lidar_fails_before_scene_load_when_echo_identity_is_unreliable(
    monkeypatch: pytest.MonkeyPatch,
):
    """OVRTX 0.4 must not expose multi-hit data whose EchoId channel is incorrect."""
    stage, renderer = _make_renderer(monkeypatch)
    lidar_prim = stage.GetPrimAtPath("/World/envs/env_0/Lidar")
    assert lidar_prim.GetAttribute("omni:sensor:Core:maxReturns").Set(2)
    spec = _lidar_product_spec()

    try:
        renderer.register_lidar_product(spec)
        with pytest.raises(RuntimeError, match=r"OVRTX 0\.4.*maxReturns=1.*EchoId"):
            renderer.prepare_stage(stage, num_envs=1)
    finally:
        renderer.close()


def test_non_partial_lidar_emits_at_authored_rate_under_faster_renderer_ticks(monkeypatch: pytest.MonkeyPatch):
    """A 10 Hz full scan emits every sixth 60 Hz renderer step without fabricating intermediate samples."""
    stage, renderer = _make_renderer(monkeypatch)
    lidar_prim = stage.GetPrimAtPath("/World/envs/env_0/Lidar")
    assert lidar_prim.GetAttribute("omni:sensor:Core:instantLidar").Set(False)
    assert lidar_prim.GetAttribute("omni:sensor:Core:partialOutputs").Set(False)
    spec = _lidar_product_spec()

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)

        for _ in range(5):
            renderer.advance_frame(1.0 / 60.0)
            assert renderer.read_lidar((spec.product_path,)) == {}

        renderer.advance_frame(1.0 / 60.0)
        first = renderer.read_lidar((spec.product_path,))[spec.product_path]
        first_frame_id = int(first.params["frameId"].reshape(-1)[0])
        first_timestamp_ns = int(first.params["timestampNs"].reshape(-1)[0])
        assert int(first.tensors["Counts"][0]) > 0
        assert first.metadata.scan_complete is True

        for _ in range(5):
            renderer.advance_frame(1.0 / 60.0)
            assert renderer.read_lidar((spec.product_path,)) == {}

        renderer.advance_frame(1.0 / 60.0)
        second = renderer.read_lidar((spec.product_path,))[spec.product_path]
        assert int(second.params["frameId"].reshape(-1)[0]) > first_frame_id
        assert int(second.params["timestampNs"].reshape(-1)[0]) > first_timestamp_ns
    finally:
        renderer.close()


def test_partial_lidar_marks_each_emitted_tick_as_scan_segment(monkeypatch: pytest.MonkeyPatch):
    """A non-instant partial output is available each tick but never mislabeled as a complete scan."""
    stage, renderer = _make_renderer(monkeypatch)
    lidar_prim = stage.GetPrimAtPath("/World/envs/env_0/Lidar")
    assert lidar_prim.GetAttribute("omni:sensor:Core:instantLidar").Set(False)
    assert lidar_prim.GetAttribute("omni:sensor:Core:partialOutputs").Set(True)
    spec = _lidar_product_spec()

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)

        previous_end_ns = 0
        for _ in range(2):
            renderer.advance_frame(1.0 / 60.0)
            frame = renderer.read_lidar((spec.product_path,))[spec.product_path]
            start_ns = int(frame.params["frameStartTimeStampNs"].reshape(-1)[0])
            end_ns = int(frame.params["frameEndTimeStampNs"].reshape(-1)[0])
            assert frame.metadata.partial_outputs is True
            assert frame.metadata.instant_lidar is False
            assert frame.metadata.scan_complete is False
            assert start_ns == previous_end_ns
            assert end_ns - start_ns == pytest.approx(1.0e9 / 60.0, abs=1.0)
            previous_end_ns = end_ns
    finally:
        renderer.close()


def test_newton_transform_sync_moves_dynamic_occluder_before_next_lidar_frame(monkeypatch: pytest.MonkeyPatch):
    """A Newton body pose update changes the next OVRTX LiDAR occlusion result."""
    target_path = "/World/envs/env_0/DynamicTarget"
    body_pose = torch.tensor(((3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0),), device="cuda:0")
    body_q = wp.from_torch(body_pose, dtype=wp.transformf)
    stage, renderer = _make_renderer(monkeypatch, body_paths=(target_path,), body_q=body_q)
    _replace_static_occluder_with_dynamic_target(stage, target_path)
    spec = _lidar_product_spec()

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        _warm_up_lidar(renderer, spec)
        occluded_frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]

        body_pose[0, :3] = body_pose.new_tensor((3.0, 5.0, 1.0))
        torch.cuda.synchronize()
        renderer.update_transforms()
        clear_frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]
        torch.cuda.synchronize()

        occluded_points = _valid_points(occluded_frame)
        clear_points = _valid_points(clear_frame)
        occluded_window = occluded_points[
            (occluded_points[:, 0] > 0.0) & (occluded_points[:, 1].abs() < 0.35) & (occluded_points[:, 2].abs() < 0.35)
        ]
        clear_window = clear_points[
            (clear_points[:, 0] > 0.0) & (clear_points[:, 1].abs() < 0.35) & (clear_points[:, 2].abs() < 0.35)
        ]
        assert len(occluded_window) > 1000, {
            "count": len(occluded_points),
            "minimum": occluded_points.amin(dim=0).tolist(),
            "maximum": occluded_points.amax(dim=0).tolist(),
        }
        assert len(clear_window) > 1000, {
            "count": len(clear_points),
            "minimum": clear_points.amin(dim=0).tolist(),
            "maximum": clear_points.amax(dim=0).tolist(),
        }
        assert float(occluded_window[:, 0].median()) == pytest.approx(1.75, abs=0.005)
        assert float(clear_window[:, 0].median()) == pytest.approx(3.95, abs=0.005)
    finally:
        renderer.close()


def test_newton_transform_sync_publishes_matrix_at_lidar_step_ordinal(monkeypatch: pytest.MonkeyPatch):
    """Newton writes the expected world matrix at the exact ordinal consumed by LiDAR."""
    target_path = "/World/envs/env_0/DynamicTarget"
    body_pose = torch.tensor(((3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0),), device="cuda:0")
    body_q = wp.from_torch(body_pose, dtype=wp.transformf)
    stage, renderer = _make_renderer(monkeypatch, body_paths=(target_path,), body_q=body_q)
    _replace_static_occluder_with_dynamic_target(stage, target_path)
    spec = _lidar_product_spec()

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        body_pose[0, :3] = body_pose.new_tensor((3.0, 5.0, 1.0))
        torch.cuda.synchronize()

        update_ordinal = renderer._current_ordinal
        renderer.update_transforms()
        _advance_and_read_lidar(renderer, (spec.product_path,))

        assert renderer._current_ordinal == update_ordinal + 1
        xforms = _read_bound_object_xforms(renderer, update_ordinal)
        np.testing.assert_allclose(xforms[0, 3, :3], (3.0, 5.0, 1.0), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(xforms[0, :3, :3], np.eye(3), rtol=0.0, atol=1.0e-12)
    finally:
        renderer.close()


def test_newton_parent_transform_moves_mounted_lidar_pose_and_sensor_frame_ranges(monkeypatch: pytest.MonkeyPatch):
    """A mounted LiDAR preserves scan endpoints while its sensor-frame ranges move."""
    mount_path = "/World/envs/env_0/SensorMount"
    body_pose = torch.tensor(((1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0),), device="cuda:0")
    body_q = wp.from_torch(body_pose, dtype=wp.transformf)
    stage, renderer = _make_renderer(monkeypatch, body_paths=(mount_path,), body_q=body_q)
    sensor_path = _move_lidar_below_mount(stage, mount_path)
    spec = _lidar_product_spec(sensor_prim_path=sensor_path)

    try:
        renderer.register_lidar_product(spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        _warm_up_lidar(renderer, spec)
        first_frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]

        body_pose[0, 0] = 2.0
        torch.cuda.synchronize()
        renderer.update_transforms()
        second_frame = _advance_and_read_lidar(renderer, (spec.product_path,))[spec.product_path]
        torch.cuda.synchronize()

        first_points = _valid_points(first_frame)
        second_points = _valid_points(second_frame)
        assert float(first_points[:, 0].max()) == pytest.approx(3.95, abs=0.03)
        assert float(second_points[:, 0].max()) == pytest.approx(2.95, abs=0.03)
        for param_name in ("frameStartPosM", "frameEndPosM"):
            torch.testing.assert_close(
                first_frame.params[param_name],
                first_frame.params[param_name].new_tensor((1.0, 2.0, 1.0)),
                rtol=0.0,
                atol=1.0e-6,
            )
        torch.testing.assert_close(
            second_frame.params["frameStartPosM"],
            second_frame.params["frameStartPosM"].new_tensor((1.0, 2.0, 1.0)),
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            second_frame.params["frameEndPosM"],
            second_frame.params["frameEndPosM"].new_tensor((2.0, 2.0, 1.0)),
            rtol=0.0,
            atol=1.0e-6,
        )
    finally:
        renderer.close()


def test_motion_compensation_removes_mounted_sensor_translation_distortion(monkeypatch: pytest.MonkeyPatch):
    """Compensated points use the scan-start pose while raw points retain intra-scan motion."""
    mount_path = "/World/envs/env_0/SensorMount"
    body_pose = torch.tensor(((1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0),), device="cuda:0")
    body_q = wp.from_torch(body_pose, dtype=wp.transformf)
    stage, renderer = _make_renderer(monkeypatch, body_paths=(mount_path,), body_q=body_q)
    noncompensated_sensor_path = _move_lidar_below_mount(stage, mount_path)
    assert stage.RemovePrim("/World/envs/env_0/OccluderPositiveX")
    compensated_sensor_path = f"{mount_path}/LidarCompensated"
    layer = stage.GetRootLayer()
    assert Sdf.CopySpec(
        layer,
        Sdf.Path(noncompensated_sensor_path),
        layer,
        Sdf.Path(compensated_sensor_path),
    )
    for sensor_path, compensation_state in (
        (noncompensated_sensor_path, "NONCOMPENSATED"),
        (compensated_sensor_path, "COMPENSATED"),
    ):
        sensor_prim = stage.GetPrimAtPath(sensor_path)
        assert sensor_prim.GetAttribute("omni:sensor:Core:instantLidar").Set(False)
        assert sensor_prim.GetAttribute("omni:sensor:Core:partialOutputs").Set(False)
        assert sensor_prim.GetAttribute("omni:sensor:Core:outputMotionCompensationState").Set(compensation_state)

    noncompensated_spec = _lidar_product_spec(
        sensor_prim_path=noncompensated_sensor_path,
        product_path="/OVRTX/Products/NonCompensated",
    )
    compensated_spec = _lidar_product_spec(
        sensor_prim_path=compensated_sensor_path,
        product_path="/OVRTX/Products/Compensated",
    )
    product_paths = (noncompensated_spec.product_path, compensated_spec.product_path)

    try:
        renderer.register_lidar_product(noncompensated_spec)
        renderer.register_lidar_product(compensated_spec)
        renderer.prepare_stage(stage, num_envs=1)
        renderer.initialize_lidar_scene(num_envs=1)
        for _ in range(3):
            _advance_and_read_lidar(renderer, product_paths)

        body_pose[0, 0] = 2.0
        torch.cuda.synchronize()
        renderer.update_transforms()
        frames = _advance_and_read_lidar(renderer, product_paths)
        torch.cuda.synchronize()

        central_x: dict[str, torch.Tensor] = {}
        for spec in (noncompensated_spec, compensated_spec):
            frame = frames[spec.product_path]
            count = int(frame.tensors["Counts"][0])
            valid = (frame.tensors["Flags"][:count] & 0x40) != 0
            points = frame.tensors["Coordinates"][:, :count].T[valid]
            offsets = frame.tensors["TimeOffsetNs"][:count][valid]
            central = points[(points[:, 0] > 0.0) & (points[:, 1].abs() < 0.35) & (points[:, 2].abs() < 0.35)]
            assert len(central) > 1000
            central_x[spec.product_path] = central[:, 0]
            assert int(offsets.min()) == 0
            assert 90_000_000 < int(offsets.max()) < 100_000_000
            torch.testing.assert_close(
                frame.params["frameStartPosM"],
                frame.params["frameStartPosM"].new_tensor((1.0, 2.0, 1.0)),
                rtol=0.0,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                frame.params["frameEndPosM"],
                frame.params["frameEndPosM"].new_tensor((2.0, 2.0, 1.0)),
                rtol=0.0,
                atol=1.0e-6,
            )

        noncompensated_x = central_x[noncompensated_spec.product_path]
        compensated_x = central_x[compensated_spec.product_path]
        assert float(noncompensated_x.min()) == pytest.approx(2.95, abs=0.01)
        assert float(noncompensated_x.max()) == pytest.approx(3.95, abs=0.01)
        assert float(noncompensated_x.max() - noncompensated_x.min()) > 0.95
        assert float(compensated_x.min()) == pytest.approx(3.95, abs=0.005)
        assert float(compensated_x.max()) == pytest.approx(3.95, abs=0.005)
        assert float(compensated_x.max() - compensated_x.min()) < 0.005
    finally:
        renderer.close()


def test_camera_and_lidar_products_share_one_initialized_renderer(monkeypatch: pytest.MonkeyPatch):
    """A camera-created shared scene retains its registered LiDAR product and renders both outputs."""
    stage, renderer = _make_renderer(monkeypatch)
    lidar_spec = _lidar_product_spec()
    camera_cfg = CameraCfg(
        height=48,
        width=64,
        prim_path="/World/envs/env_0/Camera",
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        data_types=["simple_shading_constant_diffuse"],
    )
    camera_spec = CameraRenderSpec(
        cfg=camera_cfg,
        device="cuda:0",
        num_instances=1,
        camera_prim_paths=("/World/envs/env_0/Camera",),
        view_count=1,
        camera_path_relative_to_env_0="Camera",
    )

    try:
        renderer.register_lidar_product(lidar_spec)
        renderer.prepare_stage(stage, num_envs=1)
        camera_data = CameraData.allocate(
            data_types=camera_cfg.data_types,
            height=camera_cfg.height,
            width=camera_cfg.width,
            num_views=1,
            device="cuda:0",
            supported_specs=renderer.supported_output_types(),
        )
        camera_data.create_buffers(num_views=1, device="cuda:0")
        camera_data.pos_w.torch[0].copy_(torch.tensor((1.0, 2.0, 1.0), device="cuda:0"))
        camera_data.quat_w_world.torch[0].copy_(torch.tensor((0.0, 0.0, 0.0, 1.0), device="cuda:0"))
        focal_pixels = camera_cfg.width * camera_cfg.spawn.focal_length / camera_cfg.spawn.horizontal_aperture
        camera_data.intrinsic_matrices.torch[0].copy_(
            torch.tensor(
                (
                    (focal_pixels, 0.0, camera_cfg.width * 0.5),
                    (0.0, focal_pixels, camera_cfg.height * 0.5),
                    (0.0, 0.0, 1.0),
                ),
                device="cuda:0",
            )
        )
        render_data = renderer.create_render_data(camera_spec)
        assert camera_data.output is not None
        renderer.set_outputs(render_data, camera_data.output)
        camera_output = camera_data.output["simple_shading_constant_diffuse"].torch
        camera_output.fill_(0xA5)

        renderer.update_camera(
            render_data,
            camera_data.pos_w,
            camera_data.quat_w_world,
            camera_data.intrinsic_matrices,
        )
        renderer.initialize_lidar_scene(num_envs=1)
        renderer.advance_frame(0.1)
        renderer.render(render_data)
        lidar_frame = renderer.read_lidar((lidar_spec.product_path,))[lidar_spec.product_path]
        torch.cuda.synchronize()

        image = camera_output
        assert image.shape == (1, 48, 64, 3)
        assert image.is_cuda
        assert not bool((image == 0xA5).all())
        assert 0 < int(lidar_frame.tensors["Counts"][0]) <= lidar_frame.tensors["Coordinates"].shape[1]
    finally:
        renderer.close()
