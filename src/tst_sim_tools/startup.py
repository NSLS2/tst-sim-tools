"""Script to initialize the namespace for running Bluesky."""

import os
from bluesky import RunEngine
from bluesky_tiled_plugins.writing import TiledWriter

if os.getenv("TILED") == 1:
    from tiled.client import simple

    client = simple()
else:
    from tiled.client import from_uri

    if os.getenv("TILED") == 2:
        client = from_uri("https://tiled.nsls2.bnl.gov")["tst/migration"]
    else:
        client = from_uri("https://tiled-staging.nsls2.bnl.gov")["tst/raw"]
