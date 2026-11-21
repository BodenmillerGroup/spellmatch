from importlib.util import find_spec

from pluggy import PluginManager

from . import hookspecs
from .matching.algorithms import icp, spellmatch

plugin_manager = PluginManager("spellmatch")
plugin_manager.add_hookspecs(hookspecs)
plugin_manager.load_setuptools_entrypoints("spellmatch")
plugin_manager.register(icp, name="spellmatch-icp")
if find_spec("probreg") is not None:  # optional dependency (spellmatch[probreg])
    from .matching.algorithms import probreg

    plugin_manager.register(probreg, name="spellmatch-probreg")
plugin_manager.register(spellmatch, name="spellmatch-spellmatch")
