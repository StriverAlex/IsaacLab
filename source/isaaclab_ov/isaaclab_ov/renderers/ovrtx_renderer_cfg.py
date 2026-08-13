# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

import os
import tempfile
from typing import Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height, num_envs, and data_types are obtained from the
    :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` when
    :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data` is called
    (same pattern as Isaac RTX).
    """

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    temp_usd_dir: str | None = None
    """Directory for temporary USD debug dumps written during OVRTX stage preparation.

    When set, the renderer writes ``pre_ovrtx_renderer_stage.usda`` (raw stage before
    partition attributes and export trimming) and ``ovrtx_renderer_stage.usda`` (exported
    stage plus injected render products) under this directory. Must be writable.
    """

    log_level: str = "verbose"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""

    read_gpu_transforms: bool | None = None
    """Whether OVRTX reads transforms from its internal GPU transform cache.

    ``None`` preserves the established ``ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS``
    environment setting, whose default is enabled. OVRTX 0.4 LiDAR requires this
    to be ``False`` because GPU transform propagation does not update dynamic
    LiDAR geometry reliably.
    """

    motion_bvh: Literal["disable", "enable", "auto"] | None = None
    """Motion-BVH mode used by OVRTX.

    ``None`` preserves the OVRTX Camera-only default. Time-resolved LiDAR
    configs use ``"auto"`` so moving sensors and geometry participate in the
    renderer's motion acceleration structure.
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize semantic segmentation output. Defaults to True.

    If True, semantic IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw semantic IDs are returned as an ``int32`` 1-channel array.

    Regardless of this setting, the semantic ID (or color) to label mapping is exposed via
    ``camera.data.info["semantic_segmentation"]["idToLabels"]``.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize instance segmentation output. Defaults to True.

    If True, instance IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw instance IDs are returned as a ``uint32`` 1-channel array.

    Regardless of this setting, the instance ID (or color) to prim-path mapping is exposed via
    ``camera.data.info["instance_segmentation"]["idToLabels"]`` and the instance ID (or color) to
    semantic-label mapping via ``camera.data.info["instance_segmentation"]["idToSemantics"]``.
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize instance ID segmentation output. Defaults to True.

    If True, instance IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw instance IDs are returned as a ``uint32`` 1-channel array.
    """
