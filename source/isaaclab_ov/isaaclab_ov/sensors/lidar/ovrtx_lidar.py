# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab sensor adapter for OVRTX LiDAR products."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2s

import warp as wp

from isaaclab import cloner
from isaaclab.renderers import BaseRenderer
from isaaclab.sensors import SensorBase
from isaaclab.sim import SimulationContext

from ...renderers import OVRTXRenderer
from .ovrtx_lidar_cfg import OVRTXLiDARCfg
from .ovrtx_lidar_data import OVRTXLiDARData
from .ovrtx_lidar_product import OVRTXLiDARProductSpec


def _product_path(sensor_prim_path: str) -> str:
    """Return a stable, collision-resistant USD product path for a sensor prim."""
    digest = blake2s(sensor_prim_path.encode(), digest_size=6).hexdigest()
    return f"/OVRTX/Products/Lidar_{digest}"


class OVRTXLiDAR(SensorBase):
    """Expose point clouds from authored OVRTX ``OmniLidar`` prims.

    OVRTX 0.4 cannot trace scene-partitioned geometry for PointCloud products, so this adapter currently supports
    exactly one environment.  Multi-environment initialization fails before the shared OVRTX scene is loaded.
    """

    cfg: OVRTXLiDARCfg

    def __init__(self, cfg: OVRTXLiDARCfg):
        super().__init__(cfg)
        self._product_paths: tuple[str, ...] = ()
        self._data: OVRTXLiDARData | None = None

    @property
    def product_paths(self) -> tuple[str, ...]:
        """OVRTX render products in clone-plan instance order."""
        return self._product_paths

    @property
    def data(self) -> OVRTXLiDARData:
        """Point-cloud data produced on demand."""
        self._update_outdated_buffers()
        if self._data is None:
            raise RuntimeError("OVRTXLiDAR is not initialized.")
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset sensor timing and invalidate cached PointCloud rows."""
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if self._data is None:
            return
        if env_mask is not None:
            reset_indices = tuple(
                int(index) for index in wp.to_torch(env_mask).nonzero(as_tuple=False).flatten().tolist()
            )
        elif env_ids is None:
            reset_indices = None
        else:
            reset_indices = tuple(int(index) for index in env_ids)
        self._data.reset(reset_indices)

    def prepare_renderer(self, renderer: BaseRenderer) -> None:
        """Register one non-tiled PointCloud product per cloned LiDAR prim."""
        if not isinstance(renderer, OVRTXRenderer):
            raise TypeError(f"OVRTXLiDAR requires OVRTXRenderer, got {type(renderer).__name__}.")

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("SimulationContext is not initialized.")
        plan = sim.get_clone_plan()
        if plan is None:
            raise RuntimeError("OVRTXLiDAR requires the scene ClonePlan before renderer initialization.")

        products_by_env: dict[int, OVRTXLiDARProductSpec] = {}
        for _source_root, destination, _source_path, env_ids in cloner.iter_clone_plan_matches(
            plan, self.cfg.prim_path
        ):
            suffix = cloner.get_suffix(self.cfg.prim_path, destination)
            if suffix is None:
                raise RuntimeError(
                    f"ClonePlan destination '{destination}' does not own OVRTX LiDAR path '{self.cfg.prim_path}'."
                )
            for env_id in env_ids:
                sensor_path = f"{destination.format(env_id)}{suffix}"
                existing = products_by_env.get(env_id)
                if existing is not None:
                    raise RuntimeError(
                        f"OVRTXLiDAR prim path '{self.cfg.prim_path}' matched multiple OVRTX LiDAR prims in "
                        f"environment {env_id}: '{existing.sensor_prim_path}' and '{sensor_path}'."
                    )
                products_by_env[env_id] = OVRTXLiDARProductSpec(
                    sensor_prim_path=sensor_path,
                    product_path=_product_path(sensor_path),
                    channels=self.cfg.channels,
                )

        ordered_env_ids = (
            tuple(int(env_id) for env_id in plan.env_ids.tolist())
            if plan.env_ids is not None
            else tuple(range(plan.clone_mask.shape[1]))
        )
        missing_env_ids = tuple(env_id for env_id in ordered_env_ids if env_id not in products_by_env)
        if missing_env_ids:
            raise RuntimeError(
                f"OVRTXLiDAR prim path '{self.cfg.prim_path}' is missing from environments {missing_env_ids}."
            )

        specs = tuple(products_by_env[env_id] for env_id in ordered_env_ids)
        for spec in specs:
            renderer.register_lidar_product(spec)
        self._product_paths = tuple(spec.product_path for spec in specs)

    def _initialize_impl(self) -> None:
        """Validate authored LiDAR prims and prepare the shared renderer stage."""
        super()._initialize_impl()

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("SimulationContext is not initialized.")
        renderer = sim.render_context.get_renderer(self.cfg.renderer_cfg)
        if not isinstance(renderer, OVRTXRenderer):
            raise TypeError(f"OVRTXLiDAR requires OVRTXRenderer, got {type(renderer).__name__}.")
        self._renderer = renderer
        if not self._product_paths:
            self.prepare_renderer(renderer)
        if len(self._product_paths) != self._num_envs:
            raise RuntimeError(
                f"OVRTXLiDAR registered {len(self._product_paths)} products for {self._num_envs} environments."
            )

        self._validate_authored_lidar_prims()
        sim.render_context.ensure_prepare_stage(self.stage, self._num_envs)
        renderer.initialize_lidar_scene(self._num_envs)
        self._data = OVRTXLiDARData(self._num_envs, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Populate point-cloud data from the shared renderer."""
        if self._data is None:
            raise RuntimeError("OVRTXLiDAR is not initialized.")
        if env_mask is self._ALL_ENV_MASK:
            env_indices = tuple(range(self._num_envs))
        else:
            env_indices = tuple(
                int(index) for index in wp.to_torch(env_mask).nonzero(as_tuple=False).flatten().tolist()
            )
        if not env_indices:
            return

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("SimulationContext is not initialized.")
        sim.render_context.prepare_renderer_frame(
            self._renderer,
            sim.get_physics_step_count(),
            sim.get_simulation_time(),
        )
        product_paths = tuple(self._product_paths[index] for index in env_indices)
        rendered = self._renderer.read_lidar(product_paths)
        self._data.mark_stale(env_indices)
        self._data.write_frames(
            {
                env_index: rendered[product_path]
                for env_index, product_path in zip(env_indices, product_paths)
                if product_path in rendered
            }
        )

    def _validate_authored_lidar_prims(self) -> None:
        """Require explicit Cartesian ``OmniLidar`` prims with the generic core API."""
        if self._clone_plan is None:
            raise RuntimeError("OVRTXLiDAR requires the scene ClonePlan.")
        matches = tuple(cloner.iter_clone_plan_matches(self._clone_plan, self.cfg.prim_path))
        if not matches:
            raise RuntimeError(f"OVRTXLiDAR prim path '{self.cfg.prim_path}' did not resolve through the ClonePlan.")

        for _source_root, _destination, source_path, _env_ids in matches:
            prim = self.stage.GetPrimAtPath(source_path)
            if not prim.IsValid():
                raise RuntimeError(f"OVRTX LiDAR source prim '{source_path}' does not exist.")
            if prim.GetTypeName() != "OmniLidar":
                raise RuntimeError(f"OVRTX LiDAR source prim '{source_path}' must have type OmniLidar.")
            if "OmniSensorGenericLidarCoreAPI" not in prim.GetAppliedSchemas():
                raise RuntimeError(f"OVRTX LiDAR source prim '{source_path}' must apply OmniSensorGenericLidarCoreAPI.")
            coordinates_type = prim.GetAttribute("omni:sensor:Core:elementsCoordsType").Get()
            if coordinates_type != "CARTESIAN":
                raise RuntimeError(
                    f"OVRTX LiDAR source prim '{source_path}' must author CARTESIAN coordinates, "
                    f"got {coordinates_type!r}."
                )
