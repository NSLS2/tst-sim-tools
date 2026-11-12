from importlib.metadata import version

import tst_sim_tools
from tst_sim_tools._version import __version__


def test_package_imports():
    assert tst_sim_tools.__name__ == "tst_sim_tools"


def test_version_matches_installed_metadata():
    assert __version__ == version("tst-sim-tools")
