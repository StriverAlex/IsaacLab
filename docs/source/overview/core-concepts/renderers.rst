.. _overview_renderers:

Renderers
=========

Isaac Lab uses a pluggable renderer architecture to support different rendering backends for camera sensors.
The :class:`~isaaclab.renderers.BaseRenderer` abstract base class defines the interface that all renderer
implementations must follow.

Isaac Lab supports three rendering backends:

- **Isaac RTX renderer** (``IsaacRtxRendererCfg``) — NVIDIA's Omniverse RTX rendering pipeline
  running inside Isaac Sim. Requires Isaac Sim. Best for photorealistic rendering, full camera
  sensor support (RGB, depth, semantic segmentation, etc.), and production quality outputs.
- **OVRTX renderer** (``OVRTXRendererCfg``) — A standalone RTX path-tracing renderer provided by
  the ``isaaclab_ov`` extension. Delivers RTX-quality rendering.
- **Newton Warp renderer** (``NewtonWarpRendererCfg``) — A lightweight GPU-accelerated renderer
  built on NVIDIA Warp. Works with the Newton physics backend and does **not** require Isaac Sim
  (kit-less mode). Ideal for training workflows where full RTX fidelity is not needed.

Choosing a renderer backend
----------------------------

+---------------------+-------------------------------+---------------------------------+
| Backend             | Requires Isaac Sim?           | Best For                        |
+=====================+===============================+=================================+
| Isaac RTX           | Yes                           | Full sensor fidelity, RTX       |
|                     |                               | photorealism, PhysX backend     |
+---------------------+-------------------------------+---------------------------------+
| OVRTX               | No (kit-less; needs           | RTX-quality rendering without   |
|                     | ``isaaclab_ov`` + ``ovrtx``)  | requiring Isaac Sim             |
+---------------------+-------------------------------+---------------------------------+
| Newton Warp         | No (kit-less)                 | Newton backend, fast training   |
+---------------------+-------------------------------+---------------------------------+

.. note::

   Visualization markers are not yet supported by Newton-based renderer backends,
   including the Newton Warp renderer. Use an RTX-based renderer, such as the
   Isaac RTX renderer or OVRTX renderer, when marker visualization is needed.

.. note::
   **Temporal information for camera-based RL.** Unlike RTX modes with temporal
   anti-aliasing (DLSS, DLAA, TAA), the Newton Warp renderer does not inject
   prior-frame information into the current image. Camera-control tasks that depend
   on velocity-like visual cues should add explicit temporal observations
   (e.g. task-local frame stacking) rather than relying on renderer-specific artifacts.

Architecture Overview
---------------------

The renderer system consists of:

1. **BaseRenderer** — Abstract base class defining the rendering lifecycle and interface
2. **Renderer** — Factory that instantiates the appropriate backend based on renderer configuration class
3. **RendererCfg** — Base configuration; each backend extends it with backend-specific options
4. **Concrete implementations** — Backend-specific renderers in extension packages
5. **RenderContext** — A management class for instantiating and accessing renderer instances using a **RendererCfg**.
   After instantiation, a config can then be used to acquire the instance of the renderer as needed.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   # Create a Newton Warp renderer (no Isaac Sim required)
   sim_ctx = sim_utils.SimulationContext.instance()
   # RenderContext.get_renderer will instantiate the renderer backend
   # or return an existing renderer with a matching config
   renderer: BaseRenderer = sim_ctx.render_context.get_renderer(NewtonWarpRendererCfg())
   assert isinstance(renderer, BaseRenderer)

For the RTX renderer (requires Isaac Sim):

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_physx.renderers import IsaacRtxRendererCfg

   # Create an RTX renderer
   sim_ctx = sim_utils.SimulationContext.instance()
   # RenderContext.get_renderer will instantiate the renderer backend
   # or return an existing renderer with a matching config
   renderer: BaseRenderer = sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())

For RTX renderer settings, see
:doc:`/source/how-to/configure_rendering`.

Core concepts
-------------

- **Use the RenderContext**: Always instantiate renderers via the RenderContext with a renderer-specific config class
  (e.g. ``sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())``). Do not import or instantiate concrete backend classes
  (e.g. ``IsaacRtxRenderer``, ``OVRTXRenderer``) directly—their names and package locations are
  implementation details and may change without notice.

- **Lightweight config imports**: Importing a renderer configuration class does not pull in backend-specific
  dependencies. The backend is lazily loaded when the renderer is instantiated, and instantiation may fail
  if the backend is not installed.

  .. code-block:: python

     import isaaclab.sim as sim_utils
     from isaaclab.renderers import BaseRenderer
     # Lightweight: does not import OVRTX backend dependencies
     from isaaclab_ov.renderers import OVRTXRendererCfg

     # Lazily loads ovrtx when instantiated; may fail if isaaclab_ov / ovrtx is not installed
     sim_ctx = sim_utils.SimulationContext.instance()
     renderer: BaseRenderer = sim_ctx.render_context.get_renderer(OVRTXRendererCfg())

Installing the OVRTX renderer
------------------------------

The OVRTX renderer is provided by the ``isaaclab_ov`` extension. The extension's
source package ships with the core install, but the renderer's ``ovrtx`` runtime
wheel (the `ovrtx <https://github.com/NVIDIA-Omniverse/ovrtx>`_ package, hosted on
``pypi.nvidia.com``) is **not** installed by default. You must request it
explicitly — OVRTX does **not** require Isaac Sim.

Install via the Isaac Lab CLI using the ``ov[ovrtx]`` token:

.. code-block:: bash

   # Install the ovrtx runtime wheel on top of an existing install
   ./isaaclab.sh -i ov[ovrtx]

.. note::

   The bare ``ov`` token does **not** install any runtime wheel (the source
   packages are already part of the core install). Use ``ov[ovrtx]`` (or ``ov[all]``)
   to pull in the ``ovrtx`` dependency.

Or install the ``ovrtx`` runtime wheel directly with pip (note the extra index URL):

.. isaaclab-ovrtx-install::

OVRTX LiDAR
------------

The optional :class:`~isaaclab_ov.sensors.OVRTXLiDAR` adapter exposes point clouds from
USD-authored ``OmniLidar`` prims. It deliberately separates three responsibilities:

- the referenced USD asset owns the scan pattern, timing, range, and material response;
- Isaac Lab owns cloned-environment registration, lazy sensor updates, and stable batched tensors;
- OVRTX owns GPU ray tracing and the ``PointCloud`` render output.

This separation keeps LiDAR opt-in. Existing ray-caster sensors and renderer choices are unchanged,
and a project can provide vendor-specific sensor assets without adding those models to Isaac Lab core.
The adapter requires the ovstage path used by OVRTX 0.4 and registers its non-tiled render product before
the shared renderer loads its scene. Camera and LiDAR products must use equal
:class:`~isaaclab_ov.renderers.OVRTXRendererCfg` values so the
:class:`~isaaclab.renderers.RenderContext` resolves them to the same renderer.

When no LiDAR product is registered, OVRTX cameras retain their existing on-demand rendering behavior.
Registering a LiDAR product enables a renderer-wide frame transaction: the simulation clock advances the
shared OVRTX renderer once per positive simulation-time interval, and Camera and LiDAR consumers read the
outputs from that same transaction. A read never advances the shared renderer independently.

.. warning::

   OVRTX 0.4 ``PointCloud`` products do not trace geometry carrying scene-partition attributes. Consequently,
   :class:`~isaaclab_ov.sensors.OVRTXLiDAR` currently supports exactly one environment. Configuring more than
   one environment raises a ``RuntimeError`` before the shared OVRTX scene is loaded; the adapter does not create
   one renderer per environment and does not fall back to a ray-caster backend. Camera-only OVRTX scenes retain
   their existing multi-environment support.

Configure the sensor on an ``OmniLidar`` prim already present under every cloned environment:

.. code-block:: python

   from isaaclab_ov.renderers import OVRTXRendererCfg
   from isaaclab_ov.sensors import OVRTXLiDARCfg

   ovrtx_cfg = OVRTXRendererCfg(read_gpu_transforms=False, motion_bvh="auto")
   lidar_cfg = OVRTXLiDARCfg(
       prim_path="{ENV_REGEX_NS}/Robot/Lidar",
       update_period=0.1,
       renderer_cfg=ovrtx_cfg,
   )

OVRTX 0.4 does not update dynamic LiDAR geometry reliably through its internal GPU transform cache, so a LiDAR
renderer must set ``read_gpu_transforms=False``. :class:`~isaaclab_ov.sensors.OVRTXLiDARCfg` uses that value in
its default renderer config and selects ``motion_bvh="auto"`` for moving sensors and geometry. When Camera and LiDAR
share a renderer, pass the same explicit config to both sensor configs. Camera-only
:class:`~isaaclab_ov.renderers.OVRTXRendererCfg` keeps the established GPU-transform and motion-BVH defaults.

The source prim must have type ``OmniLidar``, apply ``OmniSensorGenericLidarCoreAPI``, and author
``omni:sensor:Core:elementsCoordsType = \"CARTESIAN\"``. The adapter does not synthesize a sensor
profile or silently fall back to a generic ray caster when those requirements are missing.

After the first emitted frame, :attr:`~isaaclab_ov.sensors.OVRTXLiDARData.point_cloud` has shape
``(num_envs, max_points, 3)``. Use :attr:`~isaaclab_ov.sensors.OVRTXLiDARData.valid` to mask padded or
invalid entries. Per-return channels such as intensity, time offset, emitter, channel, tick, and echo
identity are available through :attr:`~isaaclab_ov.sensors.OVRTXLiDARData.channels`; fixed-shape OVRTX
frame parameters are available through :attr:`~isaaclab_ov.sensors.OVRTXLiDARData.params` and named
timestamp and pose properties.

The adapter distinguishes cached data from newly emitted data. The
:attr:`~isaaclab_ov.sensors.OVRTXLiDARData.has_data` mask reports rows that have received a frame since
initialization or reset, while :attr:`~isaaclab_ov.sensors.OVRTXLiDARData.is_fresh` reports rows emitted by
the latest renderer transaction. With ``partialOutputs=false``, intermediate simulation steps normally
emit no PointCloud; the previous sample remains cached with ``is_fresh=false``. With
``partialOutputs=true``, each emitted frame is a scan segment and
:attr:`~isaaclab_ov.sensors.OVRTXLiDARData.scan_complete` is false. Authored coordinate-frame and
motion-compensation intent are exposed separately because OVRTX 0.4 frame parameters do not report those
settings reliably.

Reset invalidates the selected cached rows and resets OVRTX sensor history at the current simulation time.
It does not invent a time step or force a scan, so ``has_data`` remains false until positive authoritative
simulation time produces the next frame. With the supported OVRTX 0.4 configuration, ``num_envs`` is one;
the batch dimension remains explicit so consumers do not need a separate single-sensor data layout.

- **Opaque render data**: The render data object returned by :meth:`~isaaclab.renderers.BaseRenderer.create_render_data` is passed to
  subsequent renderer methods. It should be completely opaque to the caller: inspecting or modifying it
  via get/set attributes is an anti-pattern and breaks the API contract.

.. note::

   The :class:`~isaaclab.renderers.BaseRenderer` class is under active development and may change without notice.

See Also
--------

- :doc:`scene_data_providers` — how scene data flows from physics backends to renderers
- :doc:`/source/overview/core-concepts/visualization` — lightweight visualizer backends for interactive feedback
