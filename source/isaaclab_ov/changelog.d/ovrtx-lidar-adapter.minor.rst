Added
^^^^^

* Add an opt-in OVRTX LiDAR adapter with owned GPU point-cloud tensors and an explicit OVRTX 0.4
  single-environment capability gate. Multi-environment use fails before scene loading instead of returning empty
  frames, creating per-environment renderers, or falling back to another sensor backend.
* Preserve OVRTX scan cadence and authored output semantics through freshness, scan-completeness, coordinate-frame,
  motion-compensation, timestamp, and pose metadata. Camera-only rendering remains on demand; Camera and LiDAR share
  one renderer transaction when LiDAR is enabled.
* Add explicit ``OVRTXRendererCfg.read_gpu_transforms`` and ``motion_bvh`` settings so LiDAR can select OVRTX 0.4's
  required transform and motion paths without changing the established Camera-only defaults.
* Let ``OVRTXLiDARCfg`` carry an optional project-owned profile spawner and local offset so one sensor config owns
  authoring and clone-plan registration without moving vendor profiles into Isaac Lab core.
