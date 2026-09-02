import math
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import call_in_bluesky_event_loop
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import StaticFilenameProvider, StaticPathProvider, callback_on_mock_put, init_devices, set_mock_value
from tiled import catalog
from tiled.client import Context, from_context
from tiled.server.app import build_app

from tst_sim_tools.agents.energy_alignment import (
    ALIGNMENT_SCORE,
    CENTROID_RMS_ERROR,
    CENTROID_SPAN,
    FWHM,
    LATERAL_POSITION_ERROR,
    MAX_CENTROID_ERROR,
    MAX_FWHM,
    MIN_INTENSITY,
    VERTICAL_POSITION_ERROR,
    EnergyAlignmentEvalutation,
)
from tst_sim_tools.analysis.image import image_series
from tst_sim_tools.devices.detectors import XRTScreenAcquireStatus, XRTScreenDetector
from tst_sim_tools.devices.materials import XRTCrystalSi
from tst_sim_tools.devices.mirrors import XRTOpticalElement, XRTToroidMirror
from tst_sim_tools.devices.sources import XRTWiggler
from tst_sim_tools.plans.bmm import HC_EV_ANGSTROM, acquire_with_energy_scan

pytestmark = pytest.mark.integration


def test_energy_alignment_round_trip_through_hdf_and_tiled(run_engine, documents, tmp_path: Path) -> None:
    catalog_dir = tmp_path / "tiled"
    asset_dir = tmp_path / "assets"
    catalog_dir.mkdir()
    asset_dir.mkdir()
    tiled_catalog = catalog.in_memory(
        writable_storage={"filesystem": str(catalog_dir), "sql": f"duckdb:///{catalog_dir}/catalog.db"},
        readable_storage=[str(asset_dir)],
    )

    with Context.from_app(build_app(tiled_catalog)) as context:
        client = from_context(context)
        writer = TiledWriter(client, batch_size=1, validate=True)
        writer_subscription = run_engine.subscribe(writer)
        collected, collect = documents
        document_subscription = run_engine.subscribe(collect)
        path_provider = StaticPathProvider(StaticFilenameProvider("screen"), asset_dir)

        with init_devices(mock=True):
            tpw = XRTWiggler("ca://UNIT:TPW:", name="tpw")
            dcm_c1 = XRTOpticalElement("ca://UNIT:C1:", name="dcm_c1")
            dcm_c2 = XRTOpticalElement("ca://UNIT:C2:", name="dcm_c2")
            crystal = XRTCrystalSi("ca://UNIT:SI:", name="crystal")
            m2 = XRTToroidMirror("ca://UNIT:M2:", name="m2")
            screen = XRTScreenDetector("ca://UNIT:SCREEN:", "_image", path_provider, name="screen")

        set_mock_value(crystal.lattice_spacing, HC_EV_ANGSTROM / 7000.0)
        set_mock_value(dcm_c1.fixed_pitch, 0.01)
        set_mock_value(dcm_c2.fixed_pitch, -0.02)
        set_mock_value(dcm_c1.center_y, 100.0)
        set_mock_value(dcm_c1.center_z, 200.0)
        set_mock_value(screen.driver.hist_shape_height, 5)
        set_mock_value(screen.driver.hist_shape_width, 5)
        set_mock_value(screen.driver.frames_written, 0)
        run_engine(bps.mv(screen.driver.acquire_status, XRTScreenAcquireStatus.IDLE))

        expected_images = np.zeros((2, 5, 5), dtype=np.float64)
        expected_images[0, 2, 2] = 20_000.0
        expected_images[1, 2, 3] = 20_000.0

        async def write_frame(acquire: bool) -> None:
            if not acquire:
                return
            await screen.driver.acquire_status.set(XRTScreenAcquireStatus.ACQUIRING)
            energy = await tpw.max_energy.get_value()
            if np.isclose(energy, 7000.5):
                image = expected_images[0]
            elif np.isclose(energy, 7100.5):
                image = expected_images[1]
            else:
                raise AssertionError(f"Unexpected simulated acquisition energy {energy}")

            with h5py.File(asset_dir / "screen.h5", "a") as file:
                group = file.require_group("entry/data")
                dataset: h5py.Dataset
                if "data" not in group:
                    dataset = group.create_dataset(
                        "data",
                        shape=(0, 5, 5),
                        maxshape=(None, 5, 5),
                        chunks=(1, 5, 5),
                        dtype=np.float64,
                    )
                else:
                    dataset = cast(h5py.Dataset, group["data"])
                frame_index = dataset.shape[0]
                dataset.resize((frame_index + 1, 5, 5))
                dataset[frame_index] = image

            frames_written = await screen.driver.frames_written.get_value()
            set_mock_value(screen.driver.frames_written, frames_written + 1)
            await screen.driver.acquire_status.set(XRTScreenAcquireStatus.IDLE)

        suggestions = [{"_id": "trial-0", "dcm_c2-roll": 0.0, "m2-yaw": 0.0, "m2-center_x": 0.0}]
        try:
            with callback_on_mock_put(screen.driver.acquire, write_frame):
                result = run_engine(
                    acquire_with_energy_scan(
                        suggestions,
                        [dcm_c2.roll, m2.yaw, m2.center_x],
                        [screen],
                        md={"blop_correlation_uid": "corr-1"},
                        tpw=tpw,
                        dcm_c1=dcm_c1,
                        dcm_c2=dcm_c2,
                        crystal=crystal,
                        energies=(7000.0, 7100.0),
                    )
                )
        finally:
            run_engine.unsubscribe(writer_subscription)
            run_engine.unsubscribe(document_subscription)

        uid = result.plan_result
        assert uid == collected["start"][0]["uid"]
        assert len(collected["start"]) == 1
        assert len(collected["descriptor"]) == 1
        assert len(collected["event"]) == 2
        assert collected["descriptor"][0]["data_keys"]["screen_image"]["shape"] == [1, 5, 5]
        assert len(collected["stream_resource"]) == 1
        assert len(collected["stream_datum"]) == 2
        assert collected["stop"][0]["exit_status"] == "success"
        assert collected["start"][0]["blop_suggestions"] == suggestions
        assert collected["start"][0]["run_key"] == "default_acquire"
        assert collected["start"][0]["blop_correlation_uid"] == "corr-1"
        resource = collected["stream_resource"][0]
        assert resource["data_key"] == "screen_image"
        assert resource["mimetype"] == "application/x-hdf5"
        assert resource["parameters"] == {
            "chunk_shape": (1, 5, 5),
            "dataset": "/entry/data/data",
            "join_method": "stack",
        }
        assert [datum["indices"] for datum in collected["stream_datum"]] == [
            {"start": 0, "stop": 1},
            {"start": 1, "stop": 2},
        ]
        assert screen.hints == {"fields": ["screen_image"]}
        assert call_in_bluesky_event_loop(screen.driver.acquire.get_value()) is False
        assert call_in_bluesky_event_loop(screen.driver.capture.get_value()) is False

        with h5py.File(asset_dir / "screen.h5", "r") as file:
            stored_dataset = cast(h5py.Dataset, file["/entry/data/data"])
            np.testing.assert_array_equal(stored_dataset[:], expected_images)

        stored_run = client[uid]
        stored_images = stored_run["primary"]["screen_image"].read()
        assert stored_images.shape == (2, 5, 5)
        normalized_images = image_series(stored_images)
        assert normalized_images.shape == (2, 5, 5)
        np.testing.assert_array_equal(normalized_images, expected_images)

        evaluator = EnergyAlignmentEvalutation(
            client,
            image_key="screen_image",
            target_centroid=(2.0, 2.0),
            energies=(7000.0, 7100.0),
            threshold=0.0,
            blur=0.0,
        )
        outcome = evaluator(uid, suggestions)[0]
        assert outcome["_id"] == "trial-0"
        assert outcome[MIN_INTENSITY] == 20_000.0
        assert outcome[CENTROID_RMS_ERROR] == pytest.approx(math.sqrt(0.5))
        assert outcome[MAX_CENTROID_ERROR] == 1.0
        assert outcome[CENTROID_SPAN] == 1.0
        assert outcome[LATERAL_POSITION_ERROR] == pytest.approx(math.sqrt(0.5))
        assert outcome[VERTICAL_POSITION_ERROR] == 0.0
        assert outcome[FWHM] == 1.0
        assert outcome[MAX_FWHM] == 1.0
        assert outcome[ALIGNMENT_SCORE] == pytest.approx(0.8575)
