"""Shared pytest setup.

Forces matplotlib onto the non-interactive Agg backend before any test module is
imported, so no test can open a plot window and block the run. (tests/test_zx_visual.py
draws through pyzx's zx.draw, which otherwise opens a "Figure 1" window and waits for
it to be closed.) MPLBACKEND is also set so any subprocess a test starts gets Agg too.
"""
import os

os.environ["MPLBACKEND"] = "Agg"

try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use("Agg", force=True)
