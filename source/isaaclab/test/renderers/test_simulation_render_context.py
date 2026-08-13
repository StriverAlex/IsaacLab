# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~isaaclab.renderers.render_context.RenderContext`."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

import pytest

from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.render_context import RenderContext
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera.camera_data import CameraData

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")
pytest.importorskip("isaaclab_ov")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


class _FakeBackend(BaseRenderer):
    """Test double for :class:`BaseRenderer`; does not load PhysX/Newton/OV renderer classes."""

    __slots__ = (
        "_prepare_hits",
        "_update_transforms_hits",
        "_update_geometries_hits",
        "_advance_frame_hits",
        "_reset_frame_hits",
        "_event_log",
        "_close_hits",
        "_close_raises",
    )

    def __init__(
        self,
        *,
        prepare_hits: list[int] | None = None,
        update_transforms_hits: list[int] | None = None,
        update_geometries_hits: list[int] | None = None,
        advance_frame_hits: list[float] | None = None,
        reset_frame_hits: list[float] | None = None,
        event_log: list[str] | None = None,
        close_hits: list[Any] | None = None,
        close_raises: bool = False,
    ) -> None:
        super().__init__()
        self._prepare_hits = prepare_hits
        self._update_transforms_hits = update_transforms_hits
        self._update_geometries_hits = update_geometries_hits
        self._advance_frame_hits = advance_frame_hits
        self._reset_frame_hits = reset_frame_hits
        self._event_log = event_log
        self._close_hits = close_hits
        self._close_raises = close_raises

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return {}

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        if self._prepare_hits is not None:
            self._prepare_hits.append(1)

    def create_render_data(self, spec: Any) -> Any:
        return object()

    def set_outputs(self, render_data: Any, output_data: Any) -> None:
        pass

    def update_transforms(self) -> None:
        if self._update_transforms_hits is not None:
            self._update_transforms_hits.append(1)
        if self._event_log is not None:
            self._event_log.append("ut")

    def update_geometries(self) -> None:
        if self._update_geometries_hits is not None:
            self._update_geometries_hits.append(1)
        if self._event_log is not None:
            self._event_log.append("geo")

    def update_camera(self, render_data: Any, positions: Any, orientations: Any, intrinsics: Any) -> None:
        pass

    def advance_frame(self, delta_time: float) -> None:
        if self._advance_frame_hits is not None:
            self._advance_frame_hits.append(delta_time)
        if self._event_log is not None:
            self._event_log.append("advance")

    def reset_frame_transaction(self, simulation_time: float) -> None:
        if self._reset_frame_hits is not None:
            self._reset_frame_hits.append(simulation_time)

    def render(self, render_data: Any) -> None:
        if self._event_log is not None:
            self._event_log.append("render")

    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        if self._event_log is not None:
            self._event_log.append("read")

    def cleanup(self, render_data: Any) -> None:
        pass

    def close(self) -> None:
        if self._close_hits is not None:
            self._close_hits.append(self)
        if self._close_raises:
            raise RuntimeError("backend failed to close")


class _TransactionalBackend(_FakeBackend):
    """Test backend whose products share one renderer-global clock."""

    def uses_frame_transactions(self) -> bool:
        return True


def _set_entries(ctx: RenderContext, *cfg_backend_pairs: tuple[RendererCfg, BaseRenderer]) -> None:
    ctx._renderer_entries = list(cfg_backend_pairs)  # type: ignore[assignment]  # noqa: SLF001


@pytest.fixture(autouse=True)
def _patch_renderer_factory() -> Generator[None, None, None]:
    """Never construct :class:`~isaaclab.renderers.renderer.Renderer` (real backends) in this module."""

    with patch(
        "isaaclab.renderers.render_context.Renderer",
        side_effect=lambda *_args, **_kwargs: _FakeBackend(),
    ):
        yield


def test_get_renderer_returns_equal_cfg_singleton():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    r1 = ctx.get_renderer(cfg)
    r2 = ctx.get_renderer(cfg)
    assert r1 is r2


def test_get_renderer_two_different_concrete_types_coexist():
    """Different renderer_cfg concrete classes register distinct backends (no error)."""

    ctx = RenderContext()
    rtx = ctx.get_renderer(IsaacRtxRendererCfg())
    nw = ctx.get_renderer(NewtonWarpRendererCfg())
    assert rtx is not nw


def test_ensure_prepare_stage_idempotent():
    """Second ``ensure_prepare_stage`` with same args does not call ``prepare_stage`` again."""

    ctx = RenderContext()
    prepares: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(prepare_hits=prepares)))

    ctx.ensure_prepare_stage(None, 4)
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1


def test_ensure_prepare_stage_num_envs_mismatch():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend()))

    ctx.ensure_prepare_stage(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        ctx.ensure_prepare_stage(None, 8)


def test_update_scene_state_dedupes_per_physics_step():
    """All backends' scene state hooks run once per physics step index."""

    ctx = RenderContext()
    transform_hits: list[int] = []
    geometry_hits: list[int] = []
    cfg = NewtonWarpRendererCfg()
    _set_entries(
        ctx,
        (
            cfg,
            _FakeBackend(update_transforms_hits=transform_hits, update_geometries_hits=geometry_hits),
        ),
    )

    ctx.update_scene_state(1)
    ctx.update_scene_state(1)
    assert len(transform_hits) == 1
    assert len(geometry_hits) == 1

    ctx.update_scene_state(2)
    assert len(transform_hits) == 2
    assert len(geometry_hits) == 2


def test_render_into_camera_calls_update_render_read_order():
    """render_into_camera advances a renderer once, then permits repeated reads of that frame."""
    ctx = RenderContext()
    events: list[str] = []
    cfg = IsaacRtxRendererCfg()
    fake = _TransactionalBackend(event_log=events)
    _set_entries(ctx, (cfg, fake))

    rd = object()
    cam_data = CameraData()
    ctx.render_into_camera(
        cast(BaseRenderer, fake),
        rd,
        cam_data,
        physics_step_count=1,
        simulation_time=0.02,
    )
    assert events == ["ut", "geo", "advance", "render", "read"]

    ctx.render_into_camera(
        cast(BaseRenderer, fake),
        rd,
        cam_data,
        physics_step_count=1,
        simulation_time=0.02,
    )
    assert events == ["ut", "geo", "advance", "render", "read", "render", "read"]


def test_prepare_renderer_frame_runs_inputs_and_advances_from_simulation_time_once_per_step():
    """One renderer owns one global time advance even when several products consume the frame."""
    ctx = RenderContext()
    events: list[str] = []
    advances: list[float] = []
    cfg = IsaacRtxRendererCfg()
    fake = _TransactionalBackend(event_log=events, advance_frame_hits=advances)
    _set_entries(ctx, (cfg, fake))
    owner = object()
    ctx.register_frame_input(cast(BaseRenderer, fake), owner, lambda: events.append("inputs"))

    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=4, simulation_time=0.08)
    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=4, simulation_time=0.08)
    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=7, simulation_time=0.14)

    assert events == [
        "inputs",
        "ut",
        "geo",
        "advance",
        "inputs",
        "ut",
        "geo",
        "advance",
    ]
    assert advances == pytest.approx([0.08, 0.06])


def test_prepare_renderer_frame_rejects_time_rewind():
    """A shared renderer must not silently invent a delta after simulation time moves backward."""
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    fake = _TransactionalBackend()
    _set_entries(ctx, (cfg, fake))

    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=4, simulation_time=0.08)
    with pytest.raises(RuntimeError, match="moved backward"):
        ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=5, simulation_time=0.07)


def test_prepare_renderer_frame_waits_for_positive_authoritative_time():
    """Initialization and reset at the current time do not invent a renderer delta."""
    ctx = RenderContext()
    advances: list[float] = []
    fake = _TransactionalBackend(advance_frame_hits=advances)
    _set_entries(ctx, (IsaacRtxRendererCfg(), fake))

    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=0, simulation_time=0.0)
    assert advances == []

    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=1, simulation_time=0.02)
    assert advances == pytest.approx([0.02])


def test_continuous_frame_is_shared_by_eager_step_camera_and_lidar_consumers():
    """A continuous sensor step, Camera read, and LiDAR prepare share one renderer advance."""

    class ContinuousBackend(_TransactionalBackend):
        def requires_continuous_advance(self) -> bool:
            return True

    ctx = RenderContext()
    advances: list[float] = []
    fake = ContinuousBackend(advance_frame_hits=advances)
    _set_entries(ctx, (IsaacRtxRendererCfg(), fake))

    ctx.update_scene_and_advance(physics_step_count=3, simulation_time=0.06)
    ctx.render_into_camera(
        cast(BaseRenderer, fake),
        object(),
        CameraData(),
        physics_step_count=3,
        simulation_time=0.06,
    )
    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=3, simulation_time=0.06)

    assert advances == pytest.approx([0.06])


def test_stateless_renderer_does_not_require_simulation_time_to_advance():
    """Existing camera backends retain their render-on-read contract at time zero."""
    ctx = RenderContext()
    events: list[str] = []
    fake = _FakeBackend(event_log=events)
    _set_entries(ctx, (IsaacRtxRendererCfg(), fake))

    ctx.render_into_camera(
        cast(BaseRenderer, fake),
        object(),
        CameraData(),
        physics_step_count=0,
        simulation_time=0.0,
    )

    assert events == ["ut", "geo", "render", "read"]


def test_reset_stage_prepare_flag_allows_second_prepare_stage():
    """After reset_stage_prepare_flag, ensure_prepare_stage invokes prepare_stage again."""
    ctx = RenderContext()
    prepares: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(prepare_hits=prepares)))

    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1

    ctx.reset_stage_prepare_flag()
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 2


def test_reset_scene_state_cadence_allows_repeat_update_scene_state_same_step():
    """reset_scene_state_cadence clears step dedupe so the same physics_step_count can update again."""
    ctx = RenderContext()
    hits: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(update_transforms_hits=hits)))

    ctx.update_scene_state(1)
    assert len(hits) == 1
    ctx.update_scene_state(1)
    assert len(hits) == 1

    ctx.reset_scene_state_cadence(0.02)
    ctx.update_scene_state(1)
    assert len(hits) == 2


def test_transaction_reset_clears_history_without_inventing_time():
    """Reset uses backend time reset; a new frame still requires positive simulation-time progress."""
    ctx = RenderContext()
    advances: list[float] = []
    resets: list[float] = []
    fake = _TransactionalBackend(advance_frame_hits=advances, reset_frame_hits=resets)
    _set_entries(ctx, (IsaacRtxRendererCfg(), fake))

    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=1, simulation_time=0.02)
    ctx.reset_scene_state_cadence(0.02)
    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=1, simulation_time=0.02)
    ctx.prepare_renderer_frame(cast(BaseRenderer, fake), physics_step_count=2, simulation_time=0.04)

    assert resets == [0.02]
    assert advances == pytest.approx([0.02, 0.02])


def test_close_closes_every_backend_once_and_drops_them():
    """``close`` closes each registered backend exactly once and empties the context."""
    ctx = RenderContext()
    closed: list[Any] = []
    first = _FakeBackend(close_hits=closed)
    second = _FakeBackend(close_hits=closed)
    _set_entries(ctx, (IsaacRtxRendererCfg(), first), (NewtonWarpRendererCfg(), second))

    ctx.close()
    assert closed == [first, second]

    ctx.close()
    assert closed == [first, second]


def test_close_raises_only_after_every_backend_is_closed():
    """A failing backend must not strand the others, and its failure must not go unreported."""
    ctx = RenderContext()
    closed: list[Any] = []
    failing = _FakeBackend(close_hits=closed, close_raises=True)
    healthy = _FakeBackend(close_hits=closed)
    _set_entries(ctx, (IsaacRtxRendererCfg(), failing), (NewtonWarpRendererCfg(), healthy))

    with pytest.raises(RuntimeError, match="1 renderer\\(s\\) failed to close"):
        ctx.close()

    assert closed == [failing, healthy]


def test_close_resets_stage_and_step_bookkeeping():
    """After ``close`` the context holds no backend, so a later ``ensure_prepare_stage`` is an error."""
    ctx = RenderContext()
    _set_entries(ctx, (IsaacRtxRendererCfg(), _FakeBackend()))
    ctx.ensure_prepare_stage(None, 4)

    ctx.close()

    with pytest.raises(RuntimeError, match="get_renderer must be called"):
        ctx.ensure_prepare_stage(None, 4)
