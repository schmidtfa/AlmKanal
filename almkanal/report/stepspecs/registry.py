# stepspecs/registry.py
from __future__ import annotations

import importlib
import pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SettingsFn = Callable[[dict[str, Any]], dict[str, Any]]
SummarizeFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class StepSpec:
    settings_fn: SettingsFn
    summarize_fn: SummarizeFn | None = None


_REGISTRY: dict[str, StepSpec] = {}


def register_step(name: str) -> Callable[[Callable[[], StepSpec]], Callable[[], StepSpec]]:
    """
    Decorator: define a function that returns a StepSpec, and register it as `name`.
    Example:
        @register_step("Filter")
        def spec():
            return StepSpec(settings_fn=..., summarize_fn=...)
    """

    def _decorator(builder: Callable[[], StepSpec]) -> Callable[[], StepSpec]:
        spec = builder()
        if name in _REGISTRY:
            warnings.warn(f"Overriding StepSpec for '{name}'", RuntimeWarning)
        _REGISTRY[name] = spec
        return builder

    return _decorator


def register(name: str, spec: StepSpec) -> None:
    """Direct registration if you prefer not to use the decorator."""
    if name in _REGISTRY:
        warnings.warn(f"Overriding StepSpec for '{name}'", RuntimeWarning)
    _REGISTRY[name] = spec


def get_registry() -> dict[str, StepSpec]:
    """Return a shallow copy so callers can't mutate the global by accident."""
    return dict(_REGISTRY)


def load_stepspec_package(package: str) -> None:
    """
    Import all submodules of a package so their module-level registrations run.
    e.g. load_stepspec_package('almkanal.report.stepspecs')
    """
    pkg = importlib.import_module(package)
    if not hasattr(pkg, '__path__'):
        return
    prefix = pkg.__name__ + '.'
    for _, modname, _ in pkgutil.iter_modules(pkg.__path__, prefix):
        importlib.import_module(modname)


# handy key selector
def keys_selector(*keys: str) -> SettingsFn:
    return lambda info: {k: info.get(k) for k in keys if k in info}
