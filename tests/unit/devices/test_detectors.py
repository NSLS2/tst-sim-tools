from pathlib import Path
from typing import cast

import pytest
from bluesky import plans as bp
from bluesky.protocols import StreamAsset
from bluesky.run_engine import call_in_bluesky_event_loop
from event_model.documents import StreamDatum, StreamResource
from ophyd_async.core import (
    StaticFilenameProvider,
    StaticPathProvider,
    StreamResourceDataProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)

from tst_sim_tools.devices.detectors import (
    XRTScreenAcquireLogic,
    XRTScreenAcquireStatus,
    XRTScreenDetector,
    XRTScreenHDFDataLogic,
    XRTScreenIO,
)

SCREEN_IO_SUFFIXES = [
    ("acquire", "Acquire"),
    ("acquire_status", "AcquireStatus"),
    ("num_images", "NumImages"),
    ("capture", "Capture"),
    ("file_path", "FilePath"),
    ("file_name", "FileName"),
    ("frames_written", "FramesWritten"),
    ("screen_name", "name"),
    ("center_x", "center:x"),
    ("center_y", "center:y"),
    ("center_z", "center:z"),
    ("x0", "x:x"),
    ("x1", "x:y"),
    ("x2", "x:z"),
    ("z0", "z:x"),
    ("z1", "z:y"),
    ("z2", "z:z"),
    ("lim_phys_x_lmin", "limPhysX:lmin"),
    ("lim_phys_x_lmax", "limPhysX:lmax"),
    ("lim_phys_y_lmin", "limPhysY:lmin"),
    ("lim_phys_y_lmax", "limPhysY:lmax"),
    ("hist_shape_width", "histShape:width"),
    ("hist_shape_height", "histShape:height"),
]


async def collect_stream_documents(provider, index: int) -> list[StreamAsset]:
    return [document async for document in provider.make_stream_docs(index, collections_per_event=1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("attribute", "suffix"), SCREEN_IO_SUFFIXES)
async def test_screen_io_signal_sources(attribute: str, suffix: str) -> None:
    async with init_devices(mock=True):
        driver = XRTScreenIO("ca://UNIT:", name="driver")

    assert getattr(driver, attribute).source == f"mock+ca://UNIT:{suffix}"


def test_screen_acquire_status_wire_literals() -> None:
    assert {status.name: status.value for status in XRTScreenAcquireStatus} == {
        "IDLE": "Idle",
        "ACQUIRING": "Acquiring",
        "WRITING": "Writing",
        "ERROR": "Error",
    }


@pytest.mark.asyncio
async def test_hdf_data_logic_configures_provider_and_incremental_documents(tmp_path: Path) -> None:
    async with init_devices(mock=True):
        driver = XRTScreenIO("ca://UNIT:", name="driver")

    set_mock_value(driver.hist_shape_height, 2)
    set_mock_value(driver.hist_shape_width, 3)
    set_mock_value(driver.frames_written, 0)
    path_provider = StaticPathProvider(StaticFilenameProvider("screen"), tmp_path)
    logic = XRTScreenHDFDataLogic(driver, path_provider)

    provider = cast(StreamResourceDataProvider, await logic.prepare_unbounded("screen_image"))
    datakeys = await provider.make_datakeys(collections_per_event=1)

    assert await driver.file_path.get_value() == str(tmp_path)
    assert await driver.file_name.get_value() == "screen.h5"
    assert await driver.capture.get_value() is True
    assert set(datakeys) == {"screen_image"}
    assert datakeys["screen_image"] == {
        "source": provider.uri,
        "shape": [1, 2, 3],
        "dtype": "array",
        "dtype_numpy": "<f8",
        "external": "STREAM:",
    }
    assert provider.uri.endswith("/screen.h5")
    assert len(provider.resources) == 1
    assert provider.resources[0].data_key == "screen_image"
    assert provider.resources[0].shape == (2, 3)
    assert provider.resources[0].dtype_numpy == "<f8"
    assert provider.resources[0].parameters == {"dataset": "/entry/data/data", "join_method": "stack"}
    assert provider.resources[0].chunk_shape == (1, 2, 3)
    assert logic.get_hinted_fields("screen_image") == ["screen_image"]

    assert await collect_stream_documents(provider, 0) == []
    first_documents = await collect_stream_documents(provider, 1)
    assert [name for name, _ in first_documents] == ["stream_resource", "stream_datum"]
    resource = cast(StreamResource, first_documents[0][1])
    first_datum = cast(StreamDatum, first_documents[1][1])
    assert resource["data_key"] == "screen_image"
    assert resource["mimetype"] == "application/x-hdf5"
    assert resource["uri"].endswith("/screen.h5")
    assert resource["parameters"] == {
        "chunk_shape": (1, 2, 3),
        "dataset": "/entry/data/data",
        "join_method": "stack",
    }
    assert first_datum["stream_resource"] == resource["uid"]
    assert first_datum["indices"] == {"start": 0, "stop": 1}

    assert await collect_stream_documents(provider, 1) == []
    later_documents = await collect_stream_documents(provider, 3)
    assert [name for name, _ in later_documents] == ["stream_datum"]
    later_datum = cast(StreamDatum, later_documents[0][1])
    assert later_datum["stream_resource"] == resource["uid"]
    assert later_datum["indices"] == {"start": 1, "stop": 3}

    await logic.stop()
    assert await driver.capture.get_value() is False


@pytest.mark.asyncio
async def test_acquire_logic_starts_waits_for_terminal_states_and_stops() -> None:
    async with init_devices(mock=True):
        driver = XRTScreenIO("ca://UNIT:", name="driver")
    logic = XRTScreenAcquireLogic(driver)

    await logic.start_acquiring()
    assert await driver.acquire.get_value() is True

    set_mock_value(driver.acquire_status, XRTScreenAcquireStatus.IDLE)
    await logic.wait_for_idle()
    set_mock_value(driver.acquire_status, XRTScreenAcquireStatus.ERROR)
    await logic.wait_for_idle()

    await logic.ensure_stopped()
    assert await driver.acquire.get_value() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [XRTScreenAcquireStatus.ACQUIRING, XRTScreenAcquireStatus.WRITING])
async def test_acquire_logic_times_out_in_nonterminal_states(status: XRTScreenAcquireStatus, mocker) -> None:
    async with init_devices(mock=True):
        driver = XRTScreenIO("ca://UNIT:", name="driver")
    set_mock_value(driver.acquire_status, status)
    logic = XRTScreenAcquireLogic(driver)
    mocker.patch("tst_sim_tools.devices.detectors.DEFAULT_TIMEOUT", 0.01)

    with pytest.raises(ValueError, match="not in a good state"):
        await logic.wait_for_idle()


def test_screen_detector_count_reuses_provider_and_emits_incremental_streams(
    run_engine,
    documents,
    tmp_path: Path,
    mocker,
) -> None:
    path_provider = mocker.Mock(side_effect=StaticPathProvider(StaticFilenameProvider("screen"), tmp_path))
    with init_devices(mock=True):
        detector = XRTScreenDetector("ca://UNIT:", "_image", path_provider, name="screen")

    set_mock_value(detector.driver.hist_shape_height, 2)
    set_mock_value(detector.driver.hist_shape_width, 3)
    set_mock_value(detector.driver.frames_written, 0)
    set_mock_value(detector.driver.acquire_status, XRTScreenAcquireStatus.IDLE)

    async def complete_acquisition(acquire: bool) -> None:
        if acquire:
            set_mock_value(detector.driver.acquire_status, XRTScreenAcquireStatus.ACQUIRING)
            frames_written = await detector.driver.frames_written.get_value()
            set_mock_value(detector.driver.frames_written, frames_written + 1)
            set_mock_value(detector.driver.acquire_status, XRTScreenAcquireStatus.IDLE)

    collected, collect = documents
    run_engine.subscribe(collect)
    with callback_on_mock_put(detector.driver.acquire, complete_acquisition):
        result = run_engine(bp.count([detector], num=2))

    assert len(result.run_start_uids) == 1
    assert path_provider.call_count == 1
    assert len(collected["start"]) == 1
    assert len(collected["descriptor"]) == 1
    assert len(collected["event"]) == 2
    assert len(collected["stream_resource"]) == 1
    assert [datum["indices"] for datum in collected["stream_datum"]] == [
        {"start": 0, "stop": 1},
        {"start": 1, "stop": 2},
    ]
    assert {datum["stream_resource"] for datum in collected["stream_datum"]} == {collected["stream_resource"][0]["uid"]}
    assert collected["stop"][0]["exit_status"] == "success"
    assert detector.hints == {"fields": ["screen_image"]}
    assert call_in_bluesky_event_loop(detector.driver.acquire.get_value()) is False
    assert call_in_bluesky_event_loop(detector.driver.capture.get_value()) is False
