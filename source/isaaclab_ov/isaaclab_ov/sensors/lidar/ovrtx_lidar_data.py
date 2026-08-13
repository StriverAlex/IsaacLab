# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for OVRTX LiDAR point clouds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .ovrtx_lidar_product import OVRTXLiDARFrame

_VALID_POINT_FLAG = 0x40
_STRUCTURAL_CHANNELS = {"Counts", "Flags"}


class OVRTXLiDARData:
    """Stable batched tensors populated from non-tiled OVRTX PointCloud frames.

    OVRTX reports one channel-major frame per LiDAR render product. This container
    owns environment-major copies so buffers remain valid across renderer steps.
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu"):
        if num_envs < 1:
            raise ValueError(f"OVRTXLiDARData requires at least one environment, got {num_envs}.")
        self._num_envs = num_envs
        self._capacity: int | None = None
        self._channels: dict[str, torch.Tensor] = {}
        self._params: dict[str, torch.Tensor] = {}
        self._counts: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._has_data = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._is_fresh = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._scan_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._motion_compensated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._coordinate_frame: list[str | None] = [None] * num_envs

    @property
    def point_cloud(self) -> torch.Tensor | None:
        """Point coordinates with shape ``(num_envs, max_points, 3)`` [m]."""
        return self._channels.get("Coordinates")

    @property
    def counts(self) -> torch.Tensor | None:
        """Number of point entries reported per environment."""
        return self._counts

    @property
    def valid(self) -> torch.Tensor | None:
        """Validity mask combining OVRTX ``Counts`` and the ``Flags`` valid bit."""
        return self._valid

    @property
    def channels(self) -> Mapping[str, torch.Tensor]:
        """Batched payload channels, including normalized ``Coordinates``."""
        return self._channels

    @property
    def params(self) -> Mapping[str, torch.Tensor]:
        """Batched scalar and fixed-shape PointCloud frame parameters."""
        return self._params

    @property
    def has_data(self) -> torch.Tensor:
        """Whether each row has received a frame since initialization or reset."""
        return self._has_data

    @property
    def is_fresh(self) -> torch.Tensor:
        """Whether the latest renderer transaction emitted a frame for each row."""
        return self._is_fresh

    @property
    def scan_complete(self) -> torch.Tensor:
        """Whether each row's current sample represents a complete authored scan."""
        return self._scan_complete

    @property
    def motion_compensated(self) -> torch.Tensor:
        """Authored motion-compensation intent for each current sample."""
        return self._motion_compensated

    @property
    def coordinate_frame(self) -> tuple[str | None, ...]:
        """Authored coordinate-frame token for each current sample."""
        return tuple(self._coordinate_frame)

    @property
    def frame_id(self) -> torch.Tensor | None:
        """OVRTX frame identifier for each environment."""
        return self._params.get("frameId")

    @property
    def timestamp_ns(self) -> torch.Tensor | None:
        """OVRTX frame timestamp in nanoseconds."""
        return self._params.get("timestampNs")

    @property
    def frame_start_timestamp_ns(self) -> torch.Tensor | None:
        """Beginning of the emitted scan interval in nanoseconds."""
        return self._params.get("frameStartTimeStampNs")

    @property
    def frame_end_timestamp_ns(self) -> torch.Tensor | None:
        """End of the emitted scan interval in nanoseconds."""
        return self._params.get("frameEndTimeStampNs")

    @property
    def frame_start_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Batched raw OVRTX start positions and orientations, if reported."""
        position = self._params.get("frameStartPosM")
        orientation = self._params.get("frameStartOrientation")
        if position is None or orientation is None:
            return None
        return position, orientation

    @property
    def frame_end_pose(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Batched raw OVRTX end positions and orientations, if reported."""
        position = self._params.get("frameEndPosM")
        orientation = self._params.get("frameEndOrientation")
        if position is None or orientation is None:
            return None
        return position, orientation

    def mark_stale(self, env_indices: Sequence[int]) -> None:
        """Mark selected rows as not emitted by the next completed transaction."""
        indices = self._validated_indices(env_indices)
        if indices:
            self._is_fresh[list(indices)] = False

    def reset(self, env_indices: Sequence[int] | None = None) -> None:
        """Invalidate selected cached samples without changing their allocated layout."""
        indices = tuple(range(self._num_envs)) if env_indices is None else self._validated_indices(env_indices)
        if not indices:
            return
        rows = list(indices)
        self._has_data[rows] = False
        self._is_fresh[rows] = False
        self._scan_complete[rows] = False
        self._motion_compensated[rows] = False
        for index in indices:
            self._coordinate_frame[index] = None
            for tensor in self._channels.values():
                tensor[index].zero_()
            for tensor in self._params.values():
                tensor[index].zero_()
        if self._counts is not None:
            self._counts[rows] = 0
        if self._valid is not None:
            self._valid[rows] = False

    def write_frames(self, frames: Mapping[int, OVRTXLiDARFrame]) -> None:
        """Copy owned renderer frames into their environment rows.

        Args:
            frames: Mapping from zero-based sensor instance index to its frame.

        Raises:
            ValueError: If the renderer changes the point-cloud layout after allocation.
        """
        if not frames:
            return
        first_frame = next(iter(frames.values()))
        self._validate_required_tensors(first_frame)
        capacity = int(first_frame.tensors["Coordinates"].shape[1])
        if self._capacity is None:
            self._allocate(first_frame, capacity)
        elif capacity != self._capacity:
            raise ValueError(f"OVRTX LiDAR point capacity changed from {self._capacity} to {capacity}.")

        for env_index, frame in frames.items():
            if not 0 <= env_index < self._num_envs:
                raise IndexError(f"LiDAR environment index {env_index} is outside [0, {self._num_envs}).")
            self._write_frame(env_index, frame)

    @staticmethod
    def _validate_required_tensors(frame: OVRTXLiDARFrame) -> None:
        missing = {"Coordinates", "Counts", "Flags"}.difference(frame.tensors)
        if missing:
            raise ValueError(f"OVRTX PointCloud is missing required tensors: {sorted(missing)}.")
        coordinates = frame.tensors["Coordinates"]
        if coordinates.ndim != 2 or coordinates.shape[0] != 3:
            raise ValueError(f"OVRTX Coordinates must have shape (3, max_points), got {tuple(coordinates.shape)}.")
        capacity = coordinates.shape[1]
        counts = frame.tensors["Counts"]
        flags = frame.tensors["Flags"]
        if counts.shape != (1,):
            raise ValueError(f"OVRTX Counts must have shape (1,), got {tuple(counts.shape)}.")
        if flags.shape != (capacity,):
            raise ValueError(f"OVRTX Flags must have shape ({capacity},), got {tuple(flags.shape)}.")
        for name, tensor in (("Counts", counts), ("Flags", flags)):
            if tensor.device != coordinates.device:
                raise ValueError(
                    f"OVRTX {name} device {tensor.device} does not match Coordinates device {coordinates.device}."
                )

    def _allocate(self, frame: OVRTXLiDARFrame, capacity: int) -> None:
        coordinates = frame.tensors["Coordinates"]
        if coordinates.device != self._has_data.device:
            raise ValueError(
                f"OVRTX LiDAR frame device {coordinates.device} does not match data device {self._has_data.device}."
            )
        self._capacity = capacity
        for name, tensor in frame.tensors.items():
            if name in _STRUCTURAL_CHANNELS:
                continue
            normalized = self._normalize_channel(name, tensor, capacity)
            self._channels[name] = normalized.new_zeros((self._num_envs, *normalized.shape))

        counts = frame.tensors["Counts"]
        flags = frame.tensors["Flags"]
        self._counts = counts.new_zeros(self._num_envs)
        self._valid = torch.zeros((self._num_envs, capacity), dtype=torch.bool, device=flags.device)
        for name, tensor in frame.params.items():
            self._params[name] = tensor.new_zeros((self._num_envs, *tensor.shape))

    def _write_frame(self, env_index: int, frame: OVRTXLiDARFrame) -> None:
        self._validate_required_tensors(frame)
        assert self._capacity is not None
        assert self._counts is not None
        assert self._valid is not None
        capacity = self._capacity
        if frame.tensors["Coordinates"].shape[1] != capacity:
            raise ValueError(
                f"OVRTX LiDAR point capacity changed from {capacity} to {frame.tensors['Coordinates'].shape[1]}."
            )

        payload_names = set(frame.tensors).difference(_STRUCTURAL_CHANNELS)
        if payload_names != set(self._channels):
            raise ValueError(
                "OVRTX LiDAR payload channels changed after allocation: "
                f"expected {sorted(self._channels)}, got {sorted(payload_names)}."
            )
        if set(frame.params) != set(self._params):
            raise ValueError(
                "OVRTX LiDAR frame params changed after allocation: "
                f"expected {sorted(self._params)}, got {sorted(frame.params)}."
            )

        for name, destination in self._channels.items():
            source = self._normalize_channel(name, frame.tensors[name], capacity)
            if source.shape != destination.shape[1:]:
                raise ValueError(
                    f"OVRTX LiDAR channel '{name}' changed shape from {tuple(destination.shape[1:])} "
                    f"to {tuple(source.shape)}."
                )
            if source.dtype != destination.dtype:
                raise ValueError(
                    f"OVRTX LiDAR channel '{name}' changed dtype from {destination.dtype} to {source.dtype}."
                )
            if source.device != destination.device:
                raise ValueError(
                    f"OVRTX LiDAR channel '{name}' changed device from {destination.device} to {source.device}."
                )
            destination[env_index].copy_(source)

        count = frame.tensors["Counts"].reshape(-1)[0]
        flags = frame.tensors["Flags"]
        if count.dtype != self._counts.dtype or count.device != self._counts.device:
            raise ValueError(
                "OVRTX Counts layout changed after allocation: "
                f"expected {self._counts.dtype} on {self._counts.device}, got {count.dtype} on {count.device}."
            )
        self._counts[env_index].copy_(count)
        indices = torch.arange(capacity, device=flags.device)
        self._valid[env_index].copy_((indices < count) & ((flags & _VALID_POINT_FLAG) != 0))
        for name, destination in self._params.items():
            source = frame.params[name]
            if source.shape != destination.shape[1:]:
                raise ValueError(
                    f"OVRTX LiDAR param '{name}' changed shape from {tuple(destination.shape[1:])} "
                    f"to {tuple(source.shape)}."
                )
            if source.dtype != destination.dtype or source.device != destination.device:
                raise ValueError(
                    f"OVRTX LiDAR param '{name}' changed layout from {destination.dtype} on {destination.device} "
                    f"to {source.dtype} on {source.device}."
                )
            destination[env_index].copy_(source)
        self._has_data[env_index] = True
        self._is_fresh[env_index] = True
        self._scan_complete[env_index] = frame.metadata.scan_complete
        self._motion_compensated[env_index] = frame.metadata.motion_compensated
        self._coordinate_frame[env_index] = frame.metadata.coordinate_frame

    def _validated_indices(self, env_indices: Sequence[int]) -> tuple[int, ...]:
        indices = tuple(int(index) for index in env_indices)
        invalid = tuple(index for index in indices if not 0 <= index < self._num_envs)
        if invalid:
            raise IndexError(f"LiDAR environment indices {invalid} are outside [0, {self._num_envs}).")
        return indices

    @staticmethod
    def _normalize_channel(name: str, tensor: torch.Tensor, capacity: int) -> torch.Tensor:
        if name == "Coordinates":
            return tensor.transpose(0, 1)
        if tensor.ndim == 0 or tensor.shape[0] != capacity:
            raise ValueError(
                f"OVRTX LiDAR channel '{name}' must start with point capacity {capacity}, got {tuple(tensor.shape)}."
            )
        return tensor
