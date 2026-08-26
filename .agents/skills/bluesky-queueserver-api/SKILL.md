---
name: bluesky-queueserver-api
description: Use when Python should inspect Bluesky Queue Server read-only state with bluesky-queueserver-api, REManagerAPI, status, queue_get, history_get, plans_allowed, or devices_allowed. Do not use for queue submission.
---

# Bluesky Queue Server Read-Only API

Use this skill when writing client-side Python that inspects a running Bluesky Queue Server with a read-only scoped token.

Do not use this skill to submit plans or control the queue. State-changing operations such as `item_add`, `item_add_batch`, `queue_start`, `queue_stop`, `re_pause`, `re_resume`, queue item removal, queue item reordering, lock management, or arbitrary function execution should go through the project MCP tools or another explicitly authorized control surface.

## Client Mindset

- Import `REManagerAPI` from `bluesky_queueserver_api.http` or `bluesky_queueserver_api.http.aio` only.
- Assume the HTTP endpoint is already available.
- Configure the endpoint with `http_server_uri` or `QSERVER_HTTP_SERVER_URI`.
- Configure read-only authorization with `QSERVER_READONLY_API_KEY`. Do not use a writable Queue Server token for read-only agent-side scripts.
- Do not import the beamline startup module or instantiate devices in client code; Queue Server already has the worker namespace.
- Use read-only endpoints to inspect state, discover names/signatures, map queue items to completed runs, and diagnose failures.
- Do not construct `BPlan` objects in read-only helper code; writable queue composition belongs in MCP-mediated tooling.

## Project-Local Queue Server Defaults

- Start the local Queue Server manager with `pixi run -e qs qs-server-local`.
- Start the matching HTTP server with `pixi run -e qs http-server-local`.
- The local HTTP endpoint is `http://localhost:60610`.
- The manager task points at Redis `localhost:6379`, ZMQ control `tcp://127.0.0.1:60615`, and ZMQ info `tcp://127.0.0.1:60625`.
- Permissions plus generated allowed-plan/device files are written under `/tmp/bsqs`.
- The local HTTP server is configured with single-user API key `secret`; keep reusable read-only helpers configurable via `QSERVER_READONLY_API_KEY` instead of hard-coding that writable/local key.

## Connect Over HTTP

```python
import os

from bluesky_queueserver_api.http import REManagerAPI


def make_readonly_client() -> REManagerAPI:
    """Create a read-only Queue Server HTTP client."""
    rm = REManagerAPI(
        http_server_uri=os.environ.get("QSERVER_HTTP_SERVER_URI", "http://localhost:60610"),
        request_fail_exceptions=True,
    )

    api_key = os.environ.get("QSERVER_READONLY_API_KEY")
    if api_key:
        rm.set_authorization_key(api_key=api_key)

    return rm


RM = make_readonly_client()
try:
    print(RM.status())
finally:
    RM.close()
```

## Snapshot Server State

Use this pattern at the start of analysis or queue-aware planning. It tells you whether the manager is idle, what is running, and what is waiting in the queue without changing anything.

```python
RM = make_readonly_client()
try:
    status = RM.status()
    queue = RM.queue_get(reload=True)

    print("manager_state:", status.get("manager_state"))
    print("re_state:", status.get("re_state"))
    print("running_item_uid:", status.get("running_item_uid"))
    print("queue_autostart_enabled:", status.get("queue_autostart_enabled"))
    print("running item:", queue.get("running_item") or None)
    print("queued items:", len(queue.get("items", [])))
finally:
    RM.close()
```

Fields that are commonly useful:

- `manager_state`: Queue Server manager state, such as `idle` or `executing_queue`.
- `worker_environment_exists`: whether the worker environment is open.
- `worker_environment_state`: whether the worker is idle or executing a plan.
- `re_state`: RunEngine state.
- `running_item_uid`: queue item UID currently being executed.
- `items_in_queue`: queued item count.
- `items_in_history`: history item count.
- `queue_autostart_enabled`: whether queued plans will start automatically.

## Discover Allowed Plans And Devices

Use these endpoints to understand the writable surface that the MCP tool may submit to. This is still read-only inspection.

```python
RM = make_readonly_client()
try:
    plans = RM.plans_allowed(reload=True)["plans_allowed"]
    devices = RM.devices_allowed(reload=True)["devices_allowed"]

    print("allowed plans:", sorted(plans))
    print("allowed devices:", sorted(devices))

    rel_scan = plans.get("rel_scan")
    if rel_scan:
        print("rel_scan parameters:")
        for parameter in rel_scan.get("parameters", []):
            print(parameter["name"], parameter.get("kind"), parameter.get("description", ""))

    m2 = devices.get("m2", {})
    print("m2 components:", sorted(m2.get("components", {})))
finally:
    RM.close()
```

Use the exact exposed names from these responses when asking the MCP tool to submit a plan. For ophyd-async child signals, Queue Server commonly exposes names like `m2.pitch` for submission even if metadata documents the read key as `m2-pitch`.

## Inspect Queue Contents

`queue_get()` returns the currently running item and queued items. It does not include completed items; use `history_get()` for completed queue items.

```python
RM = make_readonly_client()
try:
    queue = RM.queue_get(reload=True)

    running = queue.get("running_item") or None
    if running:
        print("running:", running["name"], running["item_uid"])
        print("args:", running.get("args", []))
        print("kwargs:", running.get("kwargs", {}))

    for index, item in enumerate(queue.get("items", []), start=1):
        print(index, item["name"], item["item_uid"])
finally:
    RM.close()
```

## Inspect History

Use history to connect submitted queue items to execution outcome and, when available, resulting Bluesky run UIDs.

Exact response shape can vary with Queue Server version and configuration, so inspect keys defensively.

```python
RM = make_readonly_client()
try:
    history = RM.history_get(reload=True)
    items = history.get("items", [])

    for item in items[-10:]:
        print("item_uid:", item.get("item_uid"))
        print("name:", item.get("name"))
        print("args:", item.get("args"))
        print("kwargs:", item.get("kwargs"))
        print("result:", item.get("result"))
        print("---")
finally:
    RM.close()
```

Helper for looking up one item:

```python
def find_history_item(rm: REManagerAPI, item_uid: str) -> dict | None:
    """Return a completed queue item by item UID, if present in history."""
    history = rm.history_get(reload=True)
    for item in history.get("items", []):
        if item.get("item_uid") == item_uid:
            return item
    return None
```

## Extract Run UIDs From History

Queue Server history may store run UIDs in different nested fields depending on version and plan result handling. Use a recursive extractor instead of assuming one exact path.

```python
from collections.abc import Iterator


def walk_values(value) -> Iterator[object]:
    """Yield nested values from dictionaries and lists."""
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from walk_values(child)


def extract_run_uids(history_item: dict) -> list[str]:
    """Extract likely Bluesky run UIDs from a history item."""
    run_uids: list[str] = []
    for value in walk_values(history_item):
        if isinstance(value, str) and len(value) == 36 and value.count("-") == 4:
            run_uids.append(value)
        elif isinstance(value, dict):
            for key in ("run_uid", "run_start", "uid"):
                candidate = value.get(key)
                if isinstance(candidate, str) and len(candidate) == 36 and candidate.count("-") == 4:
                    run_uids.append(candidate)

    return sorted(set(run_uids))
```

Use this with care: not every UUID in a history item is necessarily a run UID. Prefer explicit Queue Server fields such as `run_uids` if present.

## Poll For Completion

Polling is useful when an MCP write tool submitted an item and you need read-only Python to wait until Queue Server finishes it.

```python
import time


def wait_until_item_finishes(rm: REManagerAPI, item_uid: str, timeout: float = 300, poll_period: float = 2) -> dict | None:
    """Wait for an item to leave running/queued state and appear in history."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        queue = rm.queue_get(reload=True)
        running = queue.get("running_item") or {}
        queued_uids = {item.get("item_uid") for item in queue.get("items", [])}

        if running.get("item_uid") != item_uid and item_uid not in queued_uids:
            history_item = find_history_item(rm, item_uid)
            if history_item is not None:
                return history_item

        time.sleep(poll_period)

    raise TimeoutError(f"Queue item did not finish within {timeout} seconds: {item_uid}")
```

Prefer Queue Server status/history over grepping logs for completion. Logs are useful for diagnostics, but status/history are the API surface.

## Read Task Results Only If Created By Trusted Read-Only Helpers

If the deployment provides fixed, audited read-only helper tasks, `task_result()` may be useful for retrieving their results. Do not call arbitrary function execution from agent code.

```python
RM = make_readonly_client()
try:
    result = RM.task_result("task-uid-from-a-trusted-readonly-helper")
    print(result)
finally:
    RM.close()
```

Avoid exposing or using generic `function_execute` with read-only agent credentials unless the server strictly restricts it to audited read-only functions.

## Error Handling

- By default, rejected requests raise `RM.RequestFailedError`; catch it and inspect `ex.response` for Queue Server's message.
- Use `REManagerAPI(request_fail_exceptions=False)` if code should receive `{"success": False, "msg": ...}` responses instead of exceptions.
- `RM.RequestTimeoutError` usually means the HTTP endpoint URI is wrong, unreachable, or the request timed out.
- HTTP-specific failures may raise `RM.HTTPRequestError`, `RM.HTTPClientError`, or `RM.HTTPServerError`; client errors often mean bad auth or insufficient API scopes.
- If a read-only token is rejected for a state-changing endpoint, do not work around it; use the MCP write tool or ask for the right authorization path.

```python
try:
    status = RM.status()
except RM.RequestFailedError as ex:
    print(ex.response.get("msg", ex))
except RM.RequestTimeoutError as ex:
    print(f"Queue Server timeout: {ex}")
```

## Async Variant

Use `bluesky_queueserver_api.http.aio` and await every API call.

```python
import asyncio
import os

from bluesky_queueserver_api.http.aio import REManagerAPI


async def main():
    rm = REManagerAPI(http_server_uri=os.environ.get("QSERVER_HTTP_SERVER_URI", "http://localhost:60610"))
    api_key = os.environ.get("QSERVER_READONLY_API_KEY")
    if api_key:
        rm.set_authorization_key(api_key=api_key)

    try:
        status = await rm.status()
        queue = await rm.queue_get(reload=True)
        history = await rm.history_get(reload=True)

        print(status.get("manager_state"))
        print(len(queue.get("items", [])))
        print(len(history.get("items", [])))
    finally:
        await rm.close()


asyncio.run(main())
```

## Read-Only Checklist

Allowed for agent-side Python with a read-only token:

- `status()`
- `queue_get()`
- `history_get()`
- `plans_allowed()`
- `devices_allowed()`
- `task_result()` only for trusted read-only helper tasks

Not allowed in this skill:

- `item_add()`
- `item_add_batch()`
- `queue_start()`
- queue item modification/removal/reordering
- RunEngine pause/resume/stop/abort controls
- lock or permission mutation
- generic `function_execute()`
