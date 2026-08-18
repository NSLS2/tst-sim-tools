# AGENTS.md

## Project Shape
- Small Python package under `src/tst_sim_tools`; there are no checked-in tests or CI workflows currently.
- `src/tst_sim_tools/startup.py` is the runtime entrypoint: it creates the Bluesky `RunEngine`, subscribes a `TiledWriter`, exposes common Bluesky plans/stubs plus BMM-specific plans, and instantiates the simulated XRT devices. `BestEffortCallback` is imported for the interactive namespace but is not subscribed there.
- Device code is split by abstraction: `devices/detectors.py` contains the XRT screen detector/acquisition/HDF data logic; `devices/mirrors.py` contains optical-element, toroid, and parabolical mirror signals; `devices/materials.py`, `devices/sources.py`, and `devices/slits.py` contain silicon crystal metadata, wiggler source controls, and rectangular aperture controls.
- `plans/bmm.py` contains BMM XRT energy-change and energy-scan plans; `analysis/image.py` contains detector-image metrics; `agents/energy_alignment.py` builds a Blop agent for energy-dependent alignment.

## Environment And Commands
- Use Pixi as the source of truth; the lockfile is checked in and marked generated/binary in `.gitattributes`.
- Default/dev verification tools are dependencies, but no Pixi tasks are defined for them. Run explicit commands such as `pixi run -e dev ruff check src`, `pixi run -e dev pyright`, and `pixi run -e dev pytest`.
- There are currently no `tests/`; use focused smoke checks like `pixi run -e dev python -m compileall src` when test discovery would be empty.
- Runtime startup tasks exist only in the `startup-local` feature: `pixi run -e terminal-local start-local`, `pixi run -e terminal-local start-staging`, and `pixi run -e terminal-local start`.
- Queue Server tasks are `pixi run -e qs qs-server-local` and `pixi run -e qs http-server-local`. The manager task uses Redis at `localhost:6379`, ZMQ control/info ports `tcp://127.0.0.1:60615` and `tcp://127.0.0.1:60625`, startup script `src/tst_sim_tools/startup.py`, and permissions/device/plan output paths under `/tmp/bsqs`. The HTTP server task listens on `http://localhost:60610` with single-user API key `secret`.

## Runtime Gotchas
- `startup.py` reads `TILED` with `int(os.getenv("TILED", "0"))`; direct imports default to local Tiled storage. Use the Pixi startup tasks or set `TILED=0`, `1`, or `2` explicitly.
- `TILED=0` uses local Tiled storage at `/tmp/tst_testing`; `TILED=1` connects to `https://tiled-staging.nsls2.bnl.gov` at `tst/raw`; `TILED=2` connects to `https://tiled.nsls2.bnl.gov` at `tst/migration`.
- Keep the `_ophyd_aioca._use_pyepics_context_if_imported = lambda: None` workaround in `startup.py` unless replacing it with a verified fix; the comment says it prevents aioca/ophyd-async timeouts against caproto IOCs after RunEngine imports pyepics.
- Detector HDF resources are hard-coded to dataset `/entry/data/data`, `join_method="stack"`, and output paths from `YMDPathProvider(UUIDFilenameProvider(), PurePath("/tmp/tst_testing"))`.

## Style And Config
- Ruff uses line length 125, NumPy docstring convention, preview linting, and strict selections including docstrings, pep8-naming, pyupgrade, private imports, root logger calls, and `assert`.
- `tests/**/*` are configured to ignore `SLF001`, `S101`, and docstring rules if tests are added later.
- `src/tst_sim_tools/startup.py` intentionally ignores `F401` because it is an interactive startup namespace; do not remove imports only because they look unused there.
- Python support is `>=3.11,<3.14`, but the Pixi startup feature pins Python `>=3.12,<3.13`.
