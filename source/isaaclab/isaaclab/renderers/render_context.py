# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped renderers for camera sensors."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from isaaclab.sensors.camera.camera_data import CameraData

from .base_renderer import BaseRenderer
from .renderer import Renderer
from .renderer_cfg import RendererCfg

logger = logging.getLogger(__name__)


class RenderContext:
    """Holds :class:`BaseRenderer` instances for all :class:`Camera` sensors in a simulation.

    A camera reuses a backend when a prior camera registered a config equal under ``==`` (value
    equality) and the same concrete ``RendererCfg`` subclass. A distinct ``RendererCfg`` that
    maps to a different implementation (e.g. Isaac RTX vs Newton) produces another backend; each
    has :meth:`BaseRenderer.prepare_stage` run before use.

    :meth:`update_scene_state` is invoked at most once per :meth:`get_physics_step_count` for the
    context;
    """

    __slots__ = (
        "_renderer_entries",
        "_physics_initialized",
        "_prepared_renderer_ids",
        "_prepared_num_envs",
        "_last_scene_state_step",
        "_frame_input_updaters",
        "_last_renderer_frame_step",
        "_last_renderer_simulation_time",
    )

    def __init__(self) -> None:
        self._renderer_entries: list[tuple[RendererCfg, BaseRenderer]] = []
        self._physics_initialized: bool = False  # Set to True after the first PHYSICS_READY callback fires.
        self._prepared_renderer_ids: set[int] = set()
        self._prepared_num_envs: int | None = None
        self._last_scene_state_step: int | None = None
        self._frame_input_updaters: dict[int, dict[int, Callable[[], None]]] = {}
        self._last_renderer_frame_step: dict[int, int] = {}
        self._last_renderer_simulation_time: dict[int, float] = {}

    def _check_global_settings_compatible(self, cfg: RendererCfg) -> None:
        """Reject conflicting process-global renderer settings."""
        if getattr(cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(cfg, "global_settings"):
            return
        for stored_cfg, _renderer in self._renderer_entries:
            if getattr(stored_cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(stored_cfg, "global_settings"):
                continue
            if stored_cfg.global_settings != cfg.global_settings:
                raise ValueError(
                    "Isaac RTX global settings differ across camera renderer configs. "
                    "These settings are process-global; configure the same "
                    "IsaacRtxRendererCfg.global_settings for every Isaac RTX camera."
                )

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Return a backend for this configuration, reusing a matching instance if present.

        Lookups use ``==`` and concrete ``RendererCfg`` type, so :func:`hash` is not used (configs
        are typically not hashable).

        Args:
            cfg: Renderer configuration from the initializing camera.

        Returns:
            A shared or newly created renderer backend.
        """
        self._check_global_settings_compatible(cfg)
        for stored_cfg, r in self._renderer_entries:
            if type(stored_cfg) is type(cfg) and stored_cfg == cfg:
                return r
        new_renderer = cast(BaseRenderer, Renderer(cfg))  # type: ignore[misc]
        self._renderer_entries.append((cfg, new_renderer))
        logger.info(
            "Created new renderer for simulation: %s",
            type(new_renderer).__name__,
        )
        if self._physics_initialized:
            new_renderer.initialize()
        return new_renderer

    def ensure_initialize(self) -> None:
        """Idempotent call fired after PHYSICS_READY callback."""
        if self._physics_initialized:
            return
        self._physics_initialized = True
        for _cfg, renderer in self._renderer_entries:
            renderer.initialize()

    def ensure_prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Call :meth:`BaseRenderer.prepare_stage` for each registered backend (once per backend).

        If a new backend is added after the first :meth:`prepare_stage` call, this method ensures
        that new backend is prepared for the same ``stage`` and ``num_envs`` when the camera
        that owns it is initialized.

        Args:
            stage: USD stage passed to each backend.
            num_envs: Environment count.

        Raises:
            RuntimeError: If :meth:`get_renderer` was never called, or ``num_envs`` disagrees with
                a value already used for a prepared backend in this context.
        """
        if not self._renderer_entries:
            raise RuntimeError("get_renderer must be called at least once before ensure_prepare_stage.")
        if self._prepared_num_envs is not None and self._prepared_num_envs != num_envs:
            raise RuntimeError(
                "RenderContext prepare_stage was used with a different num_envs "
                f"({self._prepared_num_envs} vs {num_envs})."
            )
        for _cfg, renderer in self._renderer_entries:
            rid = id(renderer)
            if rid not in self._prepared_renderer_ids:
                renderer.prepare_stage(stage, num_envs)
                self._prepared_renderer_ids.add(rid)
        if self._prepared_num_envs is None:
            self._prepared_num_envs = num_envs

    def update_scene_state(self, physics_step_count: int) -> None:
        """Update scene state on all backends (at most once per step).

        Invokes :meth:`BaseRenderer.update_transforms` and then
        :meth:`BaseRenderer.update_geometries` on each registered renderer.
        """
        if not self._renderer_entries:
            return

        if self._last_scene_state_step == physics_step_count:
            return

        for _cfg, renderer in self._renderer_entries:
            renderer.update_transforms()
            renderer.update_geometries()

        self._last_scene_state_step = physics_step_count

    def register_frame_input(
        self,
        renderer: BaseRenderer,
        owner: object,
        updater: Callable[[], None],
    ) -> None:
        """Register one owner-specific input update for a shared renderer transaction."""
        self._require_registered_renderer(renderer)
        if not renderer.uses_frame_transactions():
            raise ValueError("Renderer does not use shared frame transactions.")
        self._frame_input_updaters.setdefault(id(renderer), {})[id(owner)] = updater

    def unregister_frame_input(self, renderer: BaseRenderer, owner: object) -> None:
        """Remove a previously registered renderer input update, if present."""
        updaters = self._frame_input_updaters.get(id(renderer))
        if updaters is None:
            return
        updaters.pop(id(owner), None)
        if not updaters:
            self._frame_input_updaters.pop(id(renderer), None)

    def prepare_renderer_frame(
        self,
        renderer: BaseRenderer,
        physics_step_count: int,
        simulation_time: float,
    ) -> None:
        """Stage inputs, sync the scene, and advance one shared renderer at most once per step."""
        self._require_registered_renderer(renderer)
        if not renderer.uses_frame_transactions():
            self.update_scene_state(physics_step_count)
            return
        renderer_id = id(renderer)
        if self._last_renderer_frame_step.get(renderer_id) == physics_step_count:
            return

        self._update_frame_inputs(renderer)
        self.update_scene_state(physics_step_count)
        self._advance_renderer(renderer, physics_step_count, simulation_time)

    def update_scene_and_advance(self, physics_step_count: int, simulation_time: float) -> None:
        """Sync all backends and advance every time-integrating renderer in one transaction."""
        continuous_renderers = self._continuous_renderers()
        for renderer in continuous_renderers:
            if self._last_renderer_frame_step.get(id(renderer)) != physics_step_count:
                self._update_frame_inputs(renderer)
        self.update_scene_state(physics_step_count)
        for renderer in continuous_renderers:
            self._advance_renderer(renderer, physics_step_count, simulation_time)

    def advance_continuous_renderers(self, physics_step_count: int, simulation_time: float) -> None:
        """Advance time-integrating renderers without making lazy stateless renderers eager."""
        continuous_renderers = self._continuous_renderers()
        if not continuous_renderers:
            return
        for renderer in continuous_renderers:
            if self._last_renderer_frame_step.get(id(renderer)) != physics_step_count:
                self._update_frame_inputs(renderer)
        self.update_scene_state(physics_step_count)
        for renderer in continuous_renderers:
            self._advance_renderer(renderer, physics_step_count, simulation_time)

    def _continuous_renderers(self) -> list[BaseRenderer]:
        """Return registered backends whose products integrate time between reads."""
        return [renderer for _cfg, renderer in self._renderer_entries if renderer.requires_continuous_advance()]

    def _advance_renderer(
        self,
        renderer: BaseRenderer,
        physics_step_count: int,
        simulation_time: float,
    ) -> None:
        """Advance ``renderer`` from its preceding authoritative simulation timestamp."""
        renderer_id = id(renderer)
        if self._last_renderer_frame_step.get(renderer_id) == physics_step_count:
            return
        previous_time = self._last_renderer_simulation_time.get(renderer_id, 0.0)
        if simulation_time < previous_time:
            raise RuntimeError(f"Renderer simulation time moved backward from {previous_time} to {simulation_time}.")
        delta_time = simulation_time - previous_time
        if delta_time == 0.0:
            return
        renderer.advance_frame(delta_time)
        self._last_renderer_frame_step[renderer_id] = physics_step_count
        self._last_renderer_simulation_time[renderer_id] = simulation_time

    def _update_frame_inputs(self, renderer: BaseRenderer) -> None:
        """Publish every registered sensor input before a shared renderer step."""
        for updater in self._frame_input_updaters.get(id(renderer), {}).values():
            updater()

    def _require_registered_renderer(self, renderer: BaseRenderer) -> None:
        """Reject frame operations for a backend not owned by this context."""
        if not any(candidate is renderer for _cfg, candidate in self._renderer_entries):
            raise ValueError("Renderer is not registered with this RenderContext.")

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
        simulation_time: float,
    ) -> None:
        """Sync scene state, render, and read outputs into ``camera_data``."""
        self.prepare_renderer_frame(renderer, physics_step_count, simulation_time)
        renderer.render(render_data)
        renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow :meth:`ensure_prepare_stage` to run ``prepare_stage`` again (e.g. a new USD stage)."""
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None

    def reset_scene_state_cadence(self, simulation_time: float) -> None:
        """Clear per-step dedupe and reset transactional sensor history at ``simulation_time``."""
        self._last_scene_state_step = None
        self._last_renderer_frame_step.clear()
        for _cfg, renderer in self._renderer_entries:
            if renderer.uses_frame_transactions():
                renderer.reset_frame_transaction(simulation_time)
                self._last_renderer_simulation_time[id(renderer)] = simulation_time

    def close(self) -> None:
        """Close every registered backend and drop it from this context.

        Called from :meth:`~isaaclab.sim.simulation_context.SimulationContext.clear_instance` after
        cameras have released their render data and before the stage is torn down, so
        :meth:`BaseRenderer.close` runs while the stage is still alive. A backend that raises does
        not prevent the others from closing; the failure is reported once every backend has been
        given the chance. Idempotent.

        Raises:
            RuntimeError: If any backend's :meth:`BaseRenderer.close` raised.
        """
        errors: list[Exception] = []
        for _cfg, renderer in self._renderer_entries:
            try:
                renderer.close()
            except Exception as exc:  # noqa: BLE001 - re-raised below once every backend is closed
                logger.error("Error closing renderer %s: %s", type(renderer).__name__, exc)
                errors.append(exc)
        self._renderer_entries.clear()
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None
        self._last_scene_state_step = None
        self._frame_input_updaters.clear()
        self._last_renderer_frame_step.clear()
        self._last_renderer_simulation_time.clear()
        self._physics_initialized = False

        if errors:
            # TODO: Use ExceptionGroup when ruff target-version is bumped to py311+
            raise RuntimeError(f"{len(errors)} renderer(s) failed to close") from errors[0]
