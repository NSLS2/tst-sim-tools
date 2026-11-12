# AGENTS.md

## Project Shape

- Python package under `src/tst_sim_tools`, generated from the NSLS-II Copier template `gh:NSLS2/nsls2-tools-template`; answers live in `.copier-answers.yml` (do not edit by hand). Pull template updates with `pixi run -e dev update-template` from a clean tree.
- `src/tst_sim_tools/startup.py` is the runtime entrypoint: it creates the Bluesky `RunEngine`, subscribes a `TiledWriter`, exposes common Bluesky plans/stubs plus BMM-specific plans, and instantiates the simulated XRT devices. `BestEffortCallback` is imported for the interactive namespace but is not subscribed there.
- Device code is split by abstraction: `devices/detectors.py` contains the XRT screen detector/acquisition/HDF data logic; `devices/mirrors.py` contains optical-element, toroid, and parabolical mirror signals; `devices/materials.py`, `devices/sources.py`, and `devices/slits.py` contain silicon crystal metadata, wiggler source controls, and rectangular aperture controls.
- `plans/bmm.py` contains BMM XRT energy-change and energy-scan plans; `analysis/image.py` contains detector-image metrics; `agents/energy_alignment.py` builds a Blop agent for energy-dependent alignment.

## Environment And Commands

- Use Pixi as the source of truth: the workspace is `pixi.toml` (not `[tool.pixi]` in `pyproject.toml`); the lockfile is checked in and marked generated/binary in `.gitattributes`. Solves skip packages published in the last 7 days except the `[workspace.exclude-newer.exemptions]`.
- Environments: `default` (package only), `dev` (profile + terminal + dev tools), `terminal` (interactive startup, Python 3.12), `qs` (Queue Server, Python 3.12), `py311`/`py312`/`py313` (CI matrix), and `ioc` (XRT IOC, no package install).
- Dev tasks: `pixi run -e dev lint` (pre-commit: ruff format/check, pyright, import-linter, nbstripout, mdformat), `pixi run -e dev tests` (pytest with coverage, excluding `integration`-marked tests), and `pixi run -e dev test-integration`.
- Pytest runs doctests over `src` and tests over `tests`, turns warnings into errors, blocks non-Unix sockets via pytest-socket, and ignores `src/tst_sim_tools/startup` because importing it builds the live namespace.
- Runtime startup tasks are in the `profile` feature: `pixi run -e terminal start-local`, `pixi run -e terminal start-staging`, and `pixi run -e terminal start`.
- Queue Server tasks are `pixi run -e qs qs-server-local` and `pixi run -e qs http-server-local`. The manager task uses Redis at `localhost:6379`, ZMQ control/info ports `tcp://127.0.0.1:60615` and `tcp://127.0.0.1:60625`, startup script `src/tst_sim_tools/startup/collection.py`, and permissions/device/plan output paths under `/tmp/bsqs`; `/tmp/bsqs` and `/tmp/bsqs/user_group_permissions.yaml` must exist before the manager starts. The HTTP server task listens on `http://localhost:60611` with single-user API key `secret`.
- The XRT IOC runs with `pixi run -e ioc python iocs/xrt_xml_ioc.py --xml iocs/bmm.xml --prefix 'XF:31ID1-XRT{BMM:01}'`, the prefix the startup devices expect.
- CI: `.github/workflows/ci.yml` runs pre-commit plus `tests` and `hatch build` on Python 3.11–3.13; `daily-integration-tests.yml` runs `test-integration`; `publish-pypi.yml` publishes releases via PyPI trusted publishing; `refresh-lockfile.yml` opens a monthly `pixi update` PR.

## Runtime Gotchas

- `startup/storage.py` reads `TILED` with `int(os.getenv("TILED", "0"))`; direct imports default to local Tiled storage. Use the Pixi startup tasks or set `TILED=0`, `1`, or `2` explicitly.
- `TILED=0` uses local Tiled storage at `/tmp/tst_testing`; `TILED=1` connects to `https://tiled-staging.nsls2.bnl.gov` at `tst/raw`; `TILED=2` connects to `https://tiled.nsls2.bnl.gov` at `tst/migration`.
- Detectors write their files where the chosen Tiled server reads them: `storage.data_directory` is `/tmp/tst_testing` for `TILED=0` and `/nsls2/data/tst/legacy` otherwise.
- Keep the `_ophyd_aioca._use_pyepics_context_if_imported = lambda: None` workaround in `startup.py` unless replacing it with a verified fix; the comment says it prevents aioca/ophyd-async timeouts against caproto IOCs after RunEngine imports pyepics.
- Detector HDF resources are hard-coded to dataset `/entry/data/data`, `join_method="stack"`, and output paths from `YMDPathProvider(UUIDFilenameProvider(), data_directory)`, i.e. `<data_directory>/<data key>/YYYY/MM/DD/<uuid>.h5`.
- `startup/collection.py` calls `nslsii.configure_bluesky_json_logging(ipython=get_ipython(), RE=RE)` before devices connect. It writes `<user>-<pid>.log.jsonl` and `<user>-<pid>.documents.jsonl` to `$BLUESKY_JSON_LOG_DIR` (default `/var/log/bluesky`); the directory must already exist, otherwise nslsii emits a `UserWarning` and skips the files.

## Style And Config

- Ruff uses line length 125, NumPy docstring convention, preview linting, and strict selections including docstrings, pep8-naming, pyupgrade, private imports, root logger calls, and `assert`.
- `tests/**/*` are configured to ignore `SLF001`, `S101`, and docstring rules.
- Pyright (standard mode) suppresses diagnostics under `iocs/`; the standalone caproto IOC has known type errors.
- `src/tst_sim_tools/startup/collection.py` and `startup/devices.py` intentionally ignore `F401` because they build interactive startup namespaces; do not remove imports only because they look unused there.
- Python support is `>=3.11,<3.14`, but the `terminal` and `qs` environments pin Python 3.12 through the `py312` feature.
