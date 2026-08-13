Added
^^^^^

* Add renderer preparation and shared frame-transaction hooks so time-integrating sensors can register products
  before scene loading, publish sensor poses before one renderer-wide step, and reset renderer history from the
  authoritative simulation time.
