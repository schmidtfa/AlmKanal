"""
Built-in StepSpecs live as separate modules in this package.

Typical usage in your builder:
    from almkanal.stepspecs.registry import load_stepspec_package, get_registry
    load_stepspec_package('almkanal.stepspecs')  # auto-discovers and registers all built-ins
    specs = get_registry()
"""

from .registry import StepSpec, get_registry, keys_selector, load_stepspec_package, register, register_step

__all__ = [
    'StepSpec',
    'register_step',
    'register',
    'get_registry',
    'load_stepspec_package',
    'keys_selector',
]
