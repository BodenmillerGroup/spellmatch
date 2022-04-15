from pluggy import PluginManager

from . import hookspecs
from .matching.algorithms import icp, probreg, spellmatch


plugin_manager = PluginManager("spellmatch")
plugin_manager.add_hookspecs(hookspecs)
plugin_manager.load_setuptools_entrypoints("spellmatch")
plugin_manager.register(icp, name="spellmatch-icp")
plugin_manager.register(probreg, name="spellmatch-probreg")
plugin_manager.register(spellmatch, name="spellmatch-spellmatch")
