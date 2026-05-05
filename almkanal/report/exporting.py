# almkanal/report/exporting.py
from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from pathlib import Path

from .methods_context import build_context_from_files
from .rendering import render_methods


def preprocessing_report(
    files: Sequence[str | Path],
    out_path: str | Path,
    *,
    to: str | None = None,  # e.g. "docx" | "pdf" | "html" | "md"
    template_dir: str | Path | None = None,  # override Jinja partials (optional)
    master_template: str = 'methods_master.j2',
    step_packages: Sequence[str] = ('almkanal.report.stepspecs',),
    metadata: dict[str, str] | None = None,  # Pandoc -M key=value (title, author, date, …)
    extra_args: Iterable[str] | None = None,  # extra pandoc flags (e.g., ["--pdf-engine=xelatex"])
    pandoc_path: str | None = None,  # path/command name for pandoc
) -> Path:
    """
    Render Methods text from preprocessing JSON files and convert it with Pandoc.

    Examples:
        render_methods_via_pandoc(files, "methods.docx", to="docx")
        render_methods_via_pandoc(files, "methods.pdf", to="pdf", extra_args=["--pdf-engine=xelatex"])
        render_methods_via_pandoc(files, "methods.html")  # 'to' inferred from suffix

    Returns:
        The output Path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build context and render to Markdown
    ctx = build_context_from_files(files, step_packages=step_packages)
    md = render_methods(ctx, template_dir=template_dir, master_template=master_template)

    # 2) Decide target format
    if to is None:
        to = out.suffix.lstrip('.').lower() or 'md'

    # 3) If target is Markdown, just write it out and return
    if to in {'md', 'markdown'}:
        out.write_text(md, encoding='utf-8')
        return out

    # 4) Pandoc availability
    pandoc_cmd = pandoc_path or shutil.which('pandoc')
    if not pandoc_cmd:
        raise FileNotFoundError(
            'pandoc not found. Install it (e.g., add `pandoc` to your pixi environment) '
            'or provide `pandoc_path=` explicitly.'
        )

    # 5) Assemble Pandoc command
    args = [pandoc_cmd, '-f', 'gfm', '-t', to, '-o', str(out)]
    if metadata:
        for k, v in metadata.items():
            args += ['-M', f'{k}={v}']
    if extra_args:
        args += list(extra_args)

    # 6) Run Pandoc (feed Markdown via stdin)
    #    Using a temp file is fine too; stdin keeps it simple.
    proc = subprocess.run(args, input=md.encode('utf-8'), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"pandoc failed with exit code {proc.returncode}. Command: {' '.join(args)}")

    return out
