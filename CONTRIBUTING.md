# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what
you would like to change. Please make sure to update tests as appropriate.

## Extending spellmatch

Spellmatch uses [pluggy](https://pluggy.readthedocs.io) to discover and interface with
plugins.

Currently, spellmatch supports the addition of custom mask matching algorithms. To add
your mask matching algorithm to spellmatch, create a Python package named
`spellmatch-youralgorithm`, configure your setuptools entrypoint for pluggy, and
implement the `spellmatch.hookspecs.spellmatch_get_mask_matching_algorithm` hook.

Please refer to the spellmatch reference documentation for further details.
