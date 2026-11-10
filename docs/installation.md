## Install from GitHub

Use **Python 3.11 or newer** and an environment dedicated to your project. The repository documents installation directly from GitHub:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/schmidtfa/AlmKanal.git"
```

Git must be installed and available on your command line. Package dependencies are declared in `pyproject.toml`; allow the installer to resolve them rather than installing a hand-written list of scientific packages.

## Check the installation

```bash
python -c "import almkanal; print(almkanal.__version__)"
```

Then try the [synthetic-data quick start](quickstart.md). It needs no downloaded study data, MRI, eye tracker, or audio files.

## Use Pixi

For an existing `pixi.toml`, add the dependency to its `pypi-dependencies` table:

```toml
[pypi-dependencies]
almkanal = { git = "https://github.com/schmidtfa/AlmKanal.git" }
```

When using `pyproject.toml` as your Pixi manifest, use `[tool.pixi.pypi-dependencies]` instead. The AlmKanal repository itself includes a Pixi environment and development tasks.

## Keep a reproducible environment

For a study, record the exact software revision and preserve your environment lockfile. A Git dependency can be pinned to a commit rather than following `main`. Save that identifier with your analysis scripts and outputs.

```bash
# Replace COMMIT_SHA with the commit used in your study.
python -m pip install "git+https://github.com/schmidtfa/AlmKanal.git@COMMIT_SHA"
```

## Optional report conversion

Markdown Methods output does **not** require Pandoc. Converting Methods text to other supported formats requires the **Pandoc executable** on your PATH, or an explicit `pandoc_path` argument. Installing a Python package named `pandoc` alone is not a check that the executable is available.

```bash
pandoc --version
```

PDF conversion additionally needs an appropriate PDF engine. See [report formats](methods.md#choose-an-output-format). Source reconstruction and acquisition-specific workflows may need separate anatomical data, system files, or hardware metadata; the quick start does not.
