# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the OVRTX LiDAR sensor."""

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils.configclass import configclass

from ...renderers import OVRTXRendererCfg

if TYPE_CHECKING:
    from .ovrtx_lidar import OVRTXLiDAR


@configclass
class OVRTXLiDARCfg(SensorBaseCfg):
    """Configuration for a single-environment LiDAR whose point cloud is produced by OVRTX 0.4."""

    @configclass
    class OffsetCfg:
        """Local pose of the spawned LiDAR pose frame relative to its parent."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Local translation in metres."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Local quaternion in ``(x, y, z, w)`` order."""

    class_type: type["OVRTXLiDAR"] | str = "{DIR}.ovrtx_lidar:OVRTXLiDAR"
    """Associated OVRTX LiDAR sensor class."""

    renderer_cfg: OVRTXRendererCfg = field(
        default_factory=lambda: OVRTXRendererCfg(read_gpu_transforms=False, motion_bvh="auto")
    )
    """OVRTX renderer shared with other OVRTX-backed sensors in the scene.

    The default explicitly disables GPU transform-cache reads as required by
    OVRTX 0.4 LiDAR. A Camera sharing this renderer must use an equal config.
    """

    spawn: SpawnerCfg | None = None
    """Optional profile spawner for the authored ``OmniLidar`` prim.

    When omitted, the prim must already exist in the scene. Profile attributes
    remain asset-owned; the adapter only invokes the supplied spawner.
    """

    offset: OffsetCfg = OffsetCfg()
    """Local pose passed to :attr:`spawn` when a profile is authored.

    The spawner must author this transform on the immediate parent of
    :attr:`prim_path`, not on the ``OmniLidar`` prim, whose generated USD schema
    is not ``UsdGeomXformable``. The parent must be a non-physics Xform.
    """

    channels: tuple[str, ...] = (
        "Coordinates",
        "Intensity",
        "TimeOffsetNs",
        "EmitterId",
        "ChannelId",
        "TickId",
        "EchoId",
    )
    """PointCloud channels preserved from the authored OVRTX LiDAR model."""

    def validate_config(self) -> None:
        """Validate backend-specific configuration constraints."""
        if not isinstance(self.renderer_cfg, OVRTXRendererCfg):
            raise TypeError(
                f"OVRTXLiDARCfg.renderer_cfg must be an OVRTXRendererCfg, got {type(self.renderer_cfg).__name__}."
            )
        if self.renderer_cfg.read_gpu_transforms is not False:
            raise ValueError("OVRTXLiDARCfg.renderer_cfg must set read_gpu_transforms=False for OVRTX 0.4 LiDAR.")
        if self.renderer_cfg.motion_bvh not in {"enable", "auto"}:
            raise ValueError(
                "OVRTXLiDARCfg.renderer_cfg.motion_bvh must be 'enable' or 'auto' for moving LiDAR geometry."
            )
