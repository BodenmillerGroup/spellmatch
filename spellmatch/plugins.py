from pluggy import PluginManager

from . import hookspecs
from .matching.algorithms import icp, probreg, spellmatch


def get_plugin_manager():
    pm = PluginManager("spellmatch")
    pm.add_hookspecs(hookspecs)
    pm.load_setuptools_entrypoints("spellmatch")
    pm.register(icp, name="spellmatch-icp")
    pm.register(probreg, name="spellmatch-probreg")
    pm.register(spellmatch, name="spellmatch-spellmatch")
    return pm
