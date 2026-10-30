"""Compatibility import; implementation belongs to the device package."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module('gear_sonic.devices.inspire.model')
