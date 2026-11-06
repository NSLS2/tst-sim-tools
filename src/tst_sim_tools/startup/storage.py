"""Where data goes: the Tiled server that records runs, and the directory detectors write their files to.

Chosen by the ``TILED`` environment variable: 0 (the default) a local Tiled server, 1 tiled-staging, 2 tiled.
"""

import os
from enum import IntEnum
from pathlib import Path


class TiledChoice(IntEnum):
    """Choice of tiled server to connect to."""

    SIMPLE = 0
    STAGING = 1
    PROD = 2


tiled_choice = int(os.getenv("TILED", "0"))

# Detector files must be readable by the Tiled server that records them; the local one keeps them with its catalog.
data_directory = Path("/tmp/tst_testing") if tiled_choice == TiledChoice.SIMPLE else Path("/nsls2/data/tst/legacy")
