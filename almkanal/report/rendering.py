# almkanal/report/rendering.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import (
    ChoiceLoader,
    Environment,
    FileSystemLoader,
    PackageLoader,
    StrictUndefined,
    TemplateNotFound,
)

if TYPE_CHECKING:
    from jinja2.loaders import BaseLoader  # <-- for proper loader typing


def _register_default_filters(env: Environment) -> None:
    def kv(d: dict[str, Any], omit: set[str] | None = None) -> str:
        omit = omit or set()
        items = [f'{k}={v}' for k, v in d.items() if k not in omit and v is not None]
        return '; '.join(items) if items else '—'

    env.filters['kv'] = kv


def make_env(template_dir: str | Path | None = None) -> Environment:
    """
    Loader order:
      1) user overrides in `template_dir` (if provided)
      2) packaged defaults under almkanal/report/templates/
    """
    loaders: list[BaseLoader] = []
    if template_dir is not None:
        loaders.append(FileSystemLoader(str(template_dir)))  # user overrides first
    # Use the current package ('almkanal.report') to avoid hard-coding
    loaders.append(PackageLoader(__package__, 'templates'))

    env = Environment(
        loader=ChoiceLoader(loaders),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        extensions=['jinja2.ext.do'],
    )

    _register_default_filters(env)
    return env


def render_methods(
    ctx: Any,
    *,
    template_dir: str | Path | None = None,
    master_template: str = 'methods_master.j2',
) -> str:
    env = make_env(template_dir)
    try:
        tpl = env.get_template(master_template)
    except TemplateNotFound as e:
        user_dir = str(template_dir) if template_dir is not None else '(user dir not provided)'
        raise FileNotFoundError(
            f"Template '{master_template}' not found."
            f" Checked: {user_dir} and package defaults 'almkanal/report/templates/'."
        ) from e

    return tpl.render(
        n_subjects=getattr(ctx, 'n_subjects'),
        files=getattr(ctx, 'files'),
        ordered_steps=getattr(ctx, 'ordered_steps'),
        steps=getattr(ctx, 'steps'),
    )


def render_methods_to_file(
    ctx: Any,
    out_path: str | Path,
    *,
    template_dir: str | Path | None = None,
    master_template: str = 'methods_master.j2',
    encoding: str = 'utf-8',
) -> Path:
    text = render_methods(ctx, template_dir=template_dir, master_template=master_template)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding=encoding)
    return out
