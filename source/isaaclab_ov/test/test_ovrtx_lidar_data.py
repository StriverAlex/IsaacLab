# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX LiDAR point-cloud data contract."""

import pytest
import torch
from isaaclab_ov.sensors import OVRTXLiDARData, OVRTXLiDAROutputMetadata
from isaaclab_ov.sensors.lidar.ovrtx_lidar_product import OVRTXLiDARFrame

_OUTPUT_METADATA = OVRTXLiDAROutputMetadata(
    coordinate_frame="SENSOR",
    motion_compensated=True,
    partial_outputs=True,
    instant_lidar=False,
    frame_rate_hz=10.0,
    max_returns=1,
)


def _frame(offset: float) -> OVRTXLiDARFrame:
    return OVRTXLiDARFrame(
        tensors={
            "Coordinates": torch.tensor(
                [
                    [1.0 + offset, 2.0 + offset, 30.0, 40.0],
                    [3.0 + offset, 4.0 + offset, 30.0, 40.0],
                    [5.0 + offset, 6.0 + offset, 30.0, 40.0],
                ]
            ),
            "Intensity": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "TimeOffsetNs": torch.tensor([10, 20, 30, 40], dtype=torch.int32),
            "EmitterId": torch.tensor([1, 2, 3, 4], dtype=torch.uint32),
            "Flags": torch.tensor([0x40, 0x00, 0x40, 0x40], dtype=torch.uint8),
            "Counts": torch.tensor([2], dtype=torch.int32),
        },
        params={
            "frameId": torch.tensor(7, dtype=torch.uint64),
            "timestampNs": torch.tensor(100, dtype=torch.uint64),
            "frameStartTimeStampNs": torch.tensor(90, dtype=torch.uint64),
            "frameEndTimeStampNs": torch.tensor(100, dtype=torch.uint64),
            "frameStartPosM": torch.tensor((1.0, 2.0, 3.0)),
            "frameEndPosM": torch.tensor((4.0, 5.0, 6.0)),
            "frameStartOrientation": torch.tensor((0.0, 0.0, 0.0, 1.0)),
            "frameEndOrientation": torch.tensor((0.0, 0.0, 1.0, 0.0)),
        },
        metadata=_OUTPUT_METADATA,
    )


def test_lidar_data_batches_owned_channels_and_builds_valid_mask():
    """OVRTX channel-major coordinates become padded environment-major data."""
    data = OVRTXLiDARData(num_envs=2)

    data.write_frames({0: _frame(0.0), 1: _frame(10.0)})

    assert data.point_cloud.shape == (2, 4, 3)
    torch.testing.assert_close(data.point_cloud[0, 0], torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(data.point_cloud[1, 1], torch.tensor([12.0, 14.0, 16.0]))
    torch.testing.assert_close(data.counts, torch.tensor([2, 2], dtype=torch.int32))
    torch.testing.assert_close(
        data.valid,
        torch.tensor(
            [
                [True, False, False, False],
                [True, False, False, False],
            ]
        ),
    )
    assert data.channels["EmitterId"].dtype == torch.uint32
    assert data.params["frameId"].dtype == torch.uint64
    assert data.params["frameId"].tolist() == [7, 7]


def test_lidar_data_exposes_freshness_scan_and_pose_metadata():
    """Consumers can distinguish a partial fresh frame from stale or never-received rows."""
    data = OVRTXLiDARData(num_envs=2)

    torch.testing.assert_close(data.has_data, torch.tensor([False, False]))
    torch.testing.assert_close(data.is_fresh, torch.tensor([False, False]))

    data.write_frames({0: _frame(0.0)})

    torch.testing.assert_close(data.has_data, torch.tensor([True, False]))
    torch.testing.assert_close(data.is_fresh, torch.tensor([True, False]))
    torch.testing.assert_close(data.scan_complete, torch.tensor([False, False]))
    torch.testing.assert_close(data.motion_compensated, torch.tensor([True, False]))
    assert data.coordinate_frame == ("SENSOR", None)
    assert data.timestamp_ns.tolist() == [100, 0]
    assert data.frame_start_timestamp_ns.tolist() == [90, 0]
    assert data.frame_end_timestamp_ns.tolist() == [100, 0]
    start_position, start_orientation = data.frame_start_pose
    end_position, end_orientation = data.frame_end_pose
    torch.testing.assert_close(start_position[0], torch.tensor((1.0, 2.0, 3.0)))
    torch.testing.assert_close(start_orientation[0], torch.tensor((0.0, 0.0, 0.0, 1.0)))
    torch.testing.assert_close(end_position[0], torch.tensor((4.0, 5.0, 6.0)))
    torch.testing.assert_close(end_orientation[0], torch.tensor((0.0, 0.0, 1.0, 0.0)))

    data.mark_stale((0,))
    torch.testing.assert_close(data.has_data, torch.tensor([True, False]))
    torch.testing.assert_close(data.is_fresh, torch.tensor([False, False]))

    data.reset((0,))
    torch.testing.assert_close(data.has_data, torch.tensor([False, False]))
    torch.testing.assert_close(data.counts, torch.tensor([0, 0], dtype=torch.int32))
    assert data.coordinate_frame == (None, None)


def test_lidar_data_rejects_capacity_changes_after_allocation():
    """A configured sensor has one stable output layout for RL tensor consumers."""
    data = OVRTXLiDARData(num_envs=1)
    data.write_frames({0: _frame(0.0)})
    changed = _frame(0.0)
    for name in ("Intensity", "TimeOffsetNs", "EmitterId", "Flags"):
        changed.tensors[name] = changed.tensors[name][:3]
    changed.tensors["Coordinates"] = changed.tensors["Coordinates"][:, :3]

    try:
        data.write_frames({0: changed})
    except ValueError as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("Expected a capacity change to be rejected.")


def test_lidar_data_rejects_channel_dtype_changes_after_allocation():
    """Stable RL buffers do not silently cast a changed OVRTX channel layout."""
    data = OVRTXLiDARData(num_envs=1)
    data.write_frames({0: _frame(0.0)})
    changed = _frame(0.0)
    changed.tensors["Intensity"] = changed.tensors["Intensity"].to(torch.float64)

    with pytest.raises(ValueError, match="Intensity.*dtype"):
        data.write_frames({0: changed})


def test_lidar_data_rejects_non_scalar_counts_layout():
    """One non-tiled product must report exactly one count for its sole sensor."""
    data = OVRTXLiDARData(num_envs=1)
    changed = _frame(0.0)
    changed.tensors["Counts"] = torch.tensor([2, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="Counts.*shape"):
        data.write_frames({0: changed})
