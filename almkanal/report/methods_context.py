# methods_context.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- public dataclasses for Jinja (reuse yours) ---
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from almkanal.report.stepspecs.registry import StepSpec, get_registry, load_stepspec_package


def load_jsons(files: Sequence[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for f in files:
        p = Path(f)
        with p.open('r', encoding='utf-8') as fh:
            out.append((p, json.load(fh)))
    if not out:
        raise ValueError('No JSON files provided.')
    return out


def first_info_dict(step: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    key = f'{step.lower()}_info'
    if key in payload and isinstance(payload[key], dict):
        return payload[key]
    info_keys = [k for k, v in payload.items() if k.endswith('_info') and isinstance(v, dict)]
    return payload[info_keys[0]] if len(info_keys) == 1 else payload


def deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, int | float) and isinstance(b, int | float):
        a, b = float(a), float(b)
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def require_identical(step: str, pairs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    base_name, base = pairs[0]
    for fname, d in pairs[1:]:
        if not deep_equal(base, d):
            raise ValueError(
                f"Settings mismatch in '{step}' between '{base_name}' and '{fname}'.\nBase: {base}\nOther:{d}"
            )
    return base


@dataclass
class StepContext:
    name: str
    settings: dict[str, Any]
    results: dict[str, Any]


@dataclass
class MethodsContext:
    n_subjects: int
    files: list[str]
    ordered_steps: list[str]
    steps: list[StepContext]


def build_context_from_files(
    files: Sequence[str | Path],
    *,
    step_packages: Sequence[str] = ('almkanal.report.stepspecs',),  # packages to auto-load
    extra_specs: dict[str, StepSpec] | None = None,  # programmatic overrides
) -> MethodsContext:
    # 1) load all step packages (import side-effects register specs)
    for pkg in step_packages:
        load_stepspec_package(pkg)

    # 2) compose registry (global + extras override)
    specs = get_registry()
    if extra_specs:
        specs.update(extra_specs)

    # 3) load JSONs & enforce same ordered steps
    loaded = load_jsons(files)
    first_path, first_data = loaded[0]
    ordered_steps = list(first_data.keys())
    for p, data in loaded[1:]:
        steps_here = list(data.keys())
        if steps_here != ordered_steps:
            raise ValueError(
                'Subjects do not share the same ordered steps.\n'
                f'- {first_path.name}: {ordered_steps}\n- {p.name}: {steps_here}'
            )

    # 4) gather per-steps
    per_step_pairs: dict[str, list[tuple[str, dict[str, Any]]]] = {s: [] for s in ordered_steps}
    per_step_infos: dict[str, list[dict[str, Any]]] = {s: [] for s in ordered_steps}
    for p, data in loaded:
        for step in ordered_steps:
            info = first_info_dict(step, data.get(step))
            per_step_pairs[step].append((p.name, info))
            per_step_infos[step].append(info)

    # 5) build StepContext list using specs
    steps_ctx: list[StepContext] = []
    for step in ordered_steps:
        spec = specs.get(step, StepSpec(settings_fn=lambda info: dict(info)))  # fallback: all fields
        settings_pairs = [(fn, spec.settings_fn(info)) for fn, info in per_step_pairs[step]]
        canon = require_identical(step, settings_pairs)  # enforce identical settings
        results = spec.summarize_fn(per_step_infos[step])
        steps_ctx.append(StepContext(step, canon, results))

    return MethodsContext(
        n_subjects=len(loaded),
        files=[p.name for p, _ in loaded],
        ordered_steps=ordered_steps,
        steps=steps_ctx,
    )
