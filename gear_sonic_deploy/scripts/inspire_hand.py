"""Compatibility entry point for the Inspire device package."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module('gear_sonic.devices.inspire.driver.supervisor')
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
