# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX LiDAR render-product specification."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

_PRODUCT_SCOPE = "/OVRTX/Products/"


@dataclass(frozen=True, slots=True)
class OVRTXLiDAROutputMetadata:
    """Authored scan semantics that OVRTX frame params do not report reliably."""

    coordinate_frame: str
    """Authored ``outputFrameOfReference`` token."""

    motion_compensated: bool
    """Whether the authored output requests motion compensation."""

    partial_outputs: bool
    """Whether each emitted frame may contain only the current scan segment."""

    instant_lidar: bool
    """Whether the authored LiDAR emits a complete instantaneous scan per renderer frame."""

    frame_rate_hz: float
    """Authored scan frequency in hertz."""

    max_returns: int
    """Maximum authored returns per emitter state."""

    def __post_init__(self) -> None:
        if not self.coordinate_frame:
            raise ValueError("OVRTX LiDAR coordinate_frame cannot be empty.")
        if not math.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0.0:
            raise ValueError(f"OVRTX LiDAR frame_rate_hz must be positive, got {self.frame_rate_hz}.")
        if self.max_returns < 1:
            raise ValueError(f"OVRTX LiDAR max_returns must be positive, got {self.max_returns}.")

    @property
    def scan_complete(self) -> bool:
        """Whether every emitted frame represents one complete authored scan."""
        return self.instant_lidar or not self.partial_outputs


@dataclass(frozen=True, slots=True)
class OVRTXLiDARFrame:
    """Owned tensors copied from one mapped OVRTX PointCloud frame."""

    tensors: Mapping[str, torch.Tensor]
    """Named PointCloud tensors, including auto-enabled ``Counts`` and ``Flags``."""

    params: Mapping[str, torch.Tensor]
    """Named OVRTX frame parameters copied to owned tensors on the mapped device."""

    metadata: OVRTXLiDAROutputMetadata
    """Authored output semantics associated with this product."""


@dataclass(frozen=True, slots=True)
class OVRTXLiDARProductSpec:
    """Immutable association between one LiDAR prim and one OVRTX render product."""

    sensor_prim_path: str
    """Absolute USD path of the sole LiDAR prim consumed by the product."""

    product_path: str
    """Absolute USD path assigned to the OVRTX render product."""

    channels: tuple[str, ...] = ("Coordinates", "Intensity", "TimeOffsetNs")
    """PointCloud channels requested from OVRTX."""

    def __post_init__(self) -> None:
        """Validate paths and the stable point-cloud layout."""
        if not self.sensor_prim_path:
            raise ValueError("An OVRTX LiDAR product must reference exactly one sensor prim.")
        if not self.sensor_prim_path.startswith("/"):
            raise ValueError("OVRTX LiDAR sensor_prim_path must be an absolute USD path.")
        if not self.product_path.startswith(_PRODUCT_SCOPE) or "/" in self.product_path.removeprefix(_PRODUCT_SCOPE):
            raise ValueError(f"OVRTX LiDAR product_path must be a direct child of '{_PRODUCT_SCOPE.rstrip('/')}'.")
        if "Coordinates" not in self.channels:
            raise ValueError("OVRTX LiDAR channels must include Coordinates.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("OVRTX LiDAR channels cannot contain duplicate names.")
