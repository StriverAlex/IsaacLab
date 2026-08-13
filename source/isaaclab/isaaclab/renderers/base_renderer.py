# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    def initialize(self) -> None:
        """Post-physics one-time initialization hook. Called only once."""
        return

    def requires_continuous_advance(self) -> bool:
        """Return whether this backend must advance whenever scene simulation time advances.

        Time-integrating sensors such as a non-instantaneous LiDAR override this so
        :class:`RenderContext` can drive their shared renderer independently of
        consumer read order. Camera-only and stateless renderers remain on demand.
        """
        return False

    def uses_frame_transactions(self) -> bool:
        """Return whether :meth:`advance_frame` owns this backend's global frame clock."""
        return False

    def advance_frame(self, delta_time: float) -> None:
        """Advance shared renderer time and cache product outputs for later reads.

        Stateless backends leave this as a no-op and continue doing their work in
        :meth:`render`. Backends that own a global sensor clock override it; their
        consumer-specific render/read methods must not advance that clock again.

        Args:
            delta_time: Positive simulation-time interval since this renderer's
                preceding frame transaction [s].
        """
        return

    def reset_frame_transaction(self, simulation_time: float) -> None:
        """Clear renderer-global sensor history at an authoritative simulation time."""
        return

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Pre-render per-camera setup the backend needs.

        The default implementation is a no-op. Renderer subclasses override
        to perform whatever per-camera initialization their backend requires
        — e.g. authoring stage attributes on the resolved camera prims,
        configuring per-tile GPU buffers, or any other state setup.

        Args:
            stage: Scene stage the camera prims live on, or ``None``
                when no stage context applies. Stage-less backends ignore it.
            spec: Immutable description of the tiled camera bundle.
        """
        return

    @abstractmethod
    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Per-output layout (channels + dtype) this renderer can produce.

        Outputs absent from the mapping are not produced by this backend.

        Returns:
            Mapping from supported :class:`RenderBufferKind` to its :class:`RenderBufferSpec`.
        """
        pass

    @abstractmethod
    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Prepare the stage for rendering before :meth:`create_render_data` is called.

        Some renderers need to export or preprocess the USD stage before
        creating render data. This method is called after the renderer is
        instantiated and before :meth:`create_render_data`.

        Args:
            stage: USD stage to prepare, or None if not applicable.
            num_envs: Number of environments.
        """
        pass

    @abstractmethod
    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        """Create render data for the given camera :class:`CameraRenderSpec`.

        Args:
            spec: Immutable description of the tiled camera (paths, config, device).

        Returns:
            Renderer-specific data for subsequent :meth:`render` / :meth:`read_output` calls.
        """
        pass

    @abstractmethod
    def set_outputs(self, render_data: Any, output_data: dict[str, ProxyArray]) -> None:
        """Store reference to output buffers for writing during render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_data: Dictionary mapping output names (e.g. ``"rgb"``, ``"depth"``)
                to pre-allocated :class:`~isaaclab.utils.warp.ProxyArray` wrappers where
                rendered data will be written. Use ``.warp`` for the underlying warp array
                or ``.torch`` for a zero-copy tensor view.
        """
        pass

    @abstractmethod
    def update_transforms(self) -> None:
        """Update scene transforms before rendering.

        Called to sync physics/asset pose state into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_geometries(self) -> None:
        """Update mutable geometry attributes before rendering.

        Called to sync physics-driven geometry such as mesh points, extents, or other
        per-frame geometry buffers into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_camera(
        self,
        render_data: Any,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera poses and intrinsics for the next render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            positions: Camera positions in world frame. Shape ``(N,)``, dtype ``wp.vec3f``.
                Use ``.torch`` for a ``(N, 3)`` tensor view.
            orientations: Camera orientations as quaternions ``(x, y, z, w)``. Shape ``(N,)``,
                dtype ``wp.quatf``. Use ``.torch`` for a ``(N, 4)`` tensor view.
            intrinsics: Camera intrinsic matrices. Shape ``(N,)``, dtype ``wp.mat33f``.
                Use ``.torch`` for a ``(N, 3, 3)`` tensor view.
        """
        pass

    @abstractmethod
    def render(self, render_data: Any) -> None:
        """Perform rendering and write to output buffers.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
        """
        pass

    @abstractmethod
    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        """Read rendered outputs from the renderer into the camera data container.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            camera_data: The :class:`~isaaclab.sensors.camera.camera_data.CameraData`
                instance to populate.
        """
        pass

    @abstractmethod
    def cleanup(self, render_data: Any) -> None:
        """Release renderer resources associated with the given render data.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        pass

    def close(self) -> None:
        """Release resources owned by the renderer itself rather than by a render data.

        A renderer is shared by every camera whose configuration resolves to it (see
        :meth:`~isaaclab.renderers.render_context.RenderContext.get_renderer`), so state it owns
        outlives any single camera and cannot be released from :meth:`cleanup`.
        :meth:`~isaaclab.renderers.render_context.RenderContext.close` calls this once at
        simulation teardown, while the stage and the underlying renderer backend are still alive.

        The default implementation is a no-op, for backends whose state lives entirely on the
        render data. Implementations must be idempotent.
        """
        return
