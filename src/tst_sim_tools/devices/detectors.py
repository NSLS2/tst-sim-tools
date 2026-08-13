"""Detector device abstractions."""

import asyncio
from collections.abc import Sequence
from typing import Annotated as Ann

import numpy as np
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorDataLogic,
    PathProvider,
    SignalR,
    SignalRW,
    StandardDetector,
    StreamableDataProvider,
    StreamResourceDataProvider,
    StreamResourceInfo,
    StrictEnum,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix, wait_for_good_state


class XRTScreenAcquireStatus(StrictEnum):
    """Acquisition status of an XRT screen."""

    IDLE = "Idle"
    ACQUIRING = "Acquiring"
    WRITING = "Writing"
    ERROR = "Error"


class XRTScreenIO(EpicsDevice):
    """EPICS signals for an XRT screen."""

    # --- Acquisition control ---
    acquire: Ann[SignalRW[bool], PvSuffix("Acquire")]
    acquire_status: Ann[SignalRW[XRTScreenAcquireStatus], PvSuffix("AcquireStatus")]
    num_images: Ann[SignalRW[int], PvSuffix("NumImages")]
    # TODO: image: Ann[SignalRW, PvSuffix("Image")]

    # --- File writing ---
    capture: Ann[SignalRW[bool], PvSuffix("Capture")]
    file_path: Ann[SignalRW[str], PvSuffix("FilePath")]
    file_name: Ann[SignalRW[str], PvSuffix("FileName")]
    frames_written: Ann[SignalR[int], PvSuffix("FramesWritten")]

    # -- Read-only properties ---
    screen_name: Ann[SignalR[str], PvSuffix("name")]
    center_x: Ann[SignalR[float], PvSuffix("center:x")]
    center_y: Ann[SignalR[float], PvSuffix("center:y")]
    center_z: Ann[SignalR[float], PvSuffix("center:z")]
    x0: Ann[SignalR[float], PvSuffix("x:x")]
    x1: Ann[SignalR[float], PvSuffix("x:y")]
    x2: Ann[SignalR[float], PvSuffix("x:z")]
    z0: Ann[SignalR[float], PvSuffix("z:x")]
    z1: Ann[SignalR[float], PvSuffix("z:y")]
    z2: Ann[SignalR[float], PvSuffix("z:z")]
    lim_phys_x_lmin: Ann[SignalR[float], PvSuffix("limPhysX:lmin")]
    lim_phys_x_lmax: Ann[SignalR[float], PvSuffix("limPhysX:lmax")]
    lim_phys_y_lmin: Ann[SignalR[float], PvSuffix("limPhysY:lmin")]
    lim_phys_y_lmax: Ann[SignalR[float], PvSuffix("limPhysY:lmax")]
    hist_shape_width: Ann[SignalR[int], PvSuffix("histShape:width")]
    hist_shape_height: Ann[SignalR[int], PvSuffix("histShape:height")]


class XRTScreenHDFDataLogic(DetectorDataLogic):
    """HDF file writing logic for an XRT screen."""

    def __init__(self, driver: XRTScreenIO, path_provider: PathProvider) -> None:
        self.driver = driver
        self.path_provider = path_provider

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        """Turn file capturing on."""
        path_info = self.path_provider(datakey_name)

        shape = await asyncio.gather(
            self.driver.hist_shape_height.get_value(),
            self.driver.hist_shape_width.get_value(),
        )
        await asyncio.gather(
            self.driver.file_path.set(str(path_info.directory_path)),
            self.driver.file_name.set(f"{path_info.filename}.h5"),
        )
        await self.driver.capture.set(True)

        return StreamResourceDataProvider(
            uri=f"{path_info.directory_uri}{path_info.filename}.h5",
            resources=[
                StreamResourceInfo(
                    data_key=datakey_name,
                    shape=shape,
                    dtype_numpy=np.dtype(np.float64).str,
                    parameters={"dataset": "/entry/data/data", "join_method": "stack"},
                    chunk_shape=(1, *shape),
                ),
            ],
            mimetype="application/x-hdf5",
            collections_written_signal=self.driver.frames_written,
            flush_signal=None,
        )

    async def stop(self) -> None:
        await self.driver.capture.set(False)

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        return [datakey_name]


class XRTScreenAcquireLogic(DetectorAcquireLogic):
    """Acquisition logic for an XRT screen."""

    def __init__(self, driver: XRTScreenIO) -> None:
        self.driver = driver
        self.acquire_status: AsyncStatus | None = None

    async def start_acquiring(self):
        """Start the detector acquiring."""
        await self.driver.acquire.set(True)

    async def wait_for_idle(self):
        """Wait for the detector to return to idle after the final collection."""
        if self.acquire_status:
            await self.acquire_status
        await wait_for_good_state(
            self.driver.acquire_status,
            {XRTScreenAcquireStatus.IDLE, XRTScreenAcquireStatus.ERROR},
            timeout=DEFAULT_TIMEOUT,
        )

    async def ensure_stopped(self):
        """Stop the detector and perform end-of-scan cleanup.

        Called from `unstage()`.
        """
        await self.driver.acquire.set(False)


class XRTScreenDetector(StandardDetector):
    """XRT screen as a detector."""

    def __init__(self, prefix: str, datakey_suffix: str, path_provider: PathProvider, name: str = ""):
        self.driver = XRTScreenIO(prefix)
        data_logic = XRTScreenHDFDataLogic(self.driver, path_provider)
        data_logic.datakey_suffix = datakey_suffix
        acquire_logic = XRTScreenAcquireLogic(self.driver)
        self.add_detector_logics(data_logic, acquire_logic)
        super().__init__(name)
