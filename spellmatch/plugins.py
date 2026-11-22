from importlib.util import find_spec

from pluggy import PluginManager

from . import hookspecs
from ._spellmatch import logger
from .matching.algorithms import icp, spellmatch

plugin_manager = PluginManager("spellmatch")
plugin_manager.add_hookspecs(hookspecs)
plugin_manager.load_setuptools_entrypoints("spellmatch")
plugin_manager.register(icp, name="spellmatch-icp")
if find_spec("probreg") is not None:  # optional dependency (spellmatch[probreg])
    try:
        from .matching.algorithms import probreg
    except (ImportError, OSError) as e:  # e.g. missing system libraries of open3d
        logger.warning(f"probreg algorithms are unavailable, failed to import: {e}")
    else:
        plugin_manager.register(probreg, name="spellmatch-probreg")
plugin_manager.register(spellmatch, name="spellmatch-spellmatch")
