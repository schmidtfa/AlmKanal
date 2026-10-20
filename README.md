## Welcome to the AlmKanal

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Coverage Status](https://coveralls.io/repos/github/schmidtfa/AlmKanal/badge.svg?branch=main)](https://coveralls.io/github/schmidtfa/AlmKanal?branch=main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Build readable M/EEG preprocessing pipelines in Python, check their processing order, and document what you did.**

AlmKanal is an open-source framework for modular magnetoencephalography (MEG) and electroencephalography (EEG) preprocessing, developed at the Salzburg Brain Dynamics Lab. It builds on [MNE-Python](https://mne.tools/stable/) and is inspired by the clear, composable pipeline design of [scikit-learn](https://scikit-learn.org/stable/modules/compose.html).

AlmKanal combines reusable preprocessing steps with an extensible Python API. It can be used to build a full preprocessing workflow or to apply only the next steps to data that have already been processed elsewhere. It also tracks processing settings and can generate draft text for the preprocessing section of a scientific Methods section.

> **Development status:** AlmKanal is under active development (beta). Please check the [issue tracker](https://github.com/schmidtfa/AlmKanal/issues) and validate its output for your recording system and analysis before using it in a study.

## Why AlmKanal?

- **Pipelines you can read.** Processing steps are declared in order, directly in Python, rather than hidden behind a large external configuration file. This is a design preference, not a claim that configuration-driven tools are inferior.
- **Steps you can extend.** Define a new step by subclassing `AlmKanalStep` and implementing the processing and reporting methods. Custom steps can be composed with the built-in steps.
- **Checks for incompatible step order.** Steps can declare ordering constraints; AlmKanal checks these when constructing a pipeline. For example, Maxwell filtering cannot be placed after source reconstruction in a pipeline containing both operations.
- **Start where your data are.** Run suitable steps on existing MNE `Raw` or `Epochs` objects without requiring all previous preprocessing to have been performed by AlmKanal.
- **Methods text from recorded processing.** Save processing histories as JSON and turn them into draft Methods text using Jinja-based templates. This helps connect the implemented analysis with its written description.
- **Quality control along the way.** Pipeline runs return an MNE report containing visualizations and diagnostics supplied by the steps.
- **Beyond conventional preprocessing.** The toolkit also supports temporal alignment of M/EEG recordings with VPixx eye-tracking data and with WAV audio stimuli, supporting subsequent multimodal, temporal response function (TRF), or coherence analyses.

## Installation

AlmKanal requires **Python 3.11 or newer**. It is currently installed from GitHub rather than PyPI.

With `pip`:

```bash
python -m pip install "git+https://github.com/schmidtfa/AlmKanal.git"
```

Or add the following to the `[pypi-dependencies]` section of your [Pixi](https://pixi.sh/) project manifest:

```toml
[pypi-dependencies]
almkanal = { git = "https://github.com/schmidtfa/AlmKanal.git" }
```

For reproducible analyses, pin the dependency to a specific commit or release tag. See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on setting up a development environment.

## Quick start: build a pipeline

The example below reads an MNE-compatible FIF recording, applies two steps, and saves an HTML quality-control report. Replace the input filename with your own recording.

```python
import mne

from almkanal import AlmKanal, Filter, Resample

raw = mne.io.read_raw_fif("your_recording_raw.fif", preload=True)

pipeline = AlmKanal(
    steps=[
        Filter(highpass=1.0, lowpass=40.0),
        Resample(sfreq=200),
    ]
)

processed_raw, qc_report = pipeline.run(raw)
qc_report.save("preprocessing-report.html", overwrite=True, open_browser=False)
```

The list of steps makes the processing sequence explicit. `AlmKanal` checks the supported ordering constraints when the pipeline is created, then `run()` processes the data sequentially and returns the result and an [MNE report](https://mne.tools/stable/generated/mne.Report.html).

The same pattern can be applied to previously processed `Raw` or `Epochs` data, provided the selected steps support that input type. A pipeline does **not** need to include every earlier operation, and not every step is meaningful for both data representations.

### Automatically draft a preprocessing Methods section

After running a pipeline, save its processing history and render that history into Methods text:

```python
from almkanal import preprocessing_report

pipeline.generate_json("sub-01_preprocessing.json")

preprocessing_report(
    files=["sub-01_preprocessing.json",
            "sub-02_preprocessing.json",],
    out_path="methods.md",
)
```

The resulting Markdown can be inspected and edited for a manuscript. The reporting tools also support output formats such as DOCX and HTML through [Pandoc](https://pandoc.org/).

For a study with multiple participants, `preprocessing_report()` can accept several processing-history JSON files. It checks that their recorded step sequences and relevant settings are consistent before combining them. Custom steps may require additional Methods templates to produce polished scientific prose.

**Important:** Generated text documents the processing captured by AlmKanal. It cannot reconstruct steps carried out before the data entered the pipeline, and it should always be reviewed for accuracy, completeness, and scientific appropriateness before publication.

## Define your own preprocessing step

Extending AlmKanal is intended to be straightforward. Each step inherits from `AlmKanalStep` and provides:

1. `run(data, info)`, returning a dictionary with at least a `"data"` entry and optionally additional information to record;
2. `reports(data, report, info)`, adding relevant diagnostics to the MNE report.

For example, a custom line-noise filter can be implemented as follows:

```python
import mne
from attrs import define

from almkanal import AlmKanal, AlmKanalStep, Filter


@define
class NotchAtFrequency(AlmKanalStep):
    frequency: float = 50.0
    must_be_before: tuple[str, ...] = ("SourceReconstruction",)

    def run(self, data, info):
        data.notch_filter(freqs=[self.frequency])
        return {
            "data": data,
            "notch_info": {"frequency_hz": self.frequency},
        }

    def reports(self, data, report, info):
        if isinstance(data, mne.io.BaseRaw):
            report.add_raw(data, butterfly=False, psd=True, title="After notch filtering")
        elif isinstance(data, mne.BaseEpochs):
            report.add_epochs(data, psd=False, title="After notch filtering")


pipeline = AlmKanal(
    steps=[
        NotchAtFrequency(frequency=50.0),
        Filter(highpass=1.0, lowpass=40.0),
    ]
)
```

The `must_be_before` declaration illustrates how a custom step can participate in pipeline-order checks. Constraints are evaluated when the referenced step is also present in the pipeline. Validation detects **encoded** dependencies and restrictions; it is not a substitute for checking the scientific validity of an analysis. In case you disagree with some of our `must_be_before` or `must_be_after` declarations let us know and we can adjust them if appropriate :-)

Processing metadata can also be used in the Methods generator. For custom scientific prose, see the reporting infrastructure in [`almkanal/report/`](almkanal/report/), including its step specifications and Jinja templates.

## Built-in processing and multimodal workflows

The package includes building blocks for commonly used M/EEG operations, including:

| Area | Examples |
| --- | --- |
| Sensor-level preprocessing | Filtering, resampling, referencing, Maxwell filtering, RANSAC-based bad-channel handling, ICA, and physiological-signal cleaning |
| Events and epochs | Event handling and epoch creation |
| Source-space analysis | Forward modeling, spatial filtering, and source reconstruction |
| Time-locked and multimodal analyses | Preparation of audio-aligned data for TRF workflows, along with M/EEG alignment to VPixx eye-tracking recordings and WAV stimuli |
| Reporting | Per-step QC reports, processing-history JSON, and draft preprocessing Methods text |

The [template pipelines](template_pipelines/) offer starting points for more complete analyses. Availability and input requirements vary across steps; in particular, source analysis and acquisition-specific alignment require appropriate recording metadata and study-specific inputs.

### Temporal alignment matters

AlmKanal includes functionality to align M/EEG recordings with VPixx eye-tracking data and to relate neural recordings to external WAV stimuli. Such alignment is particularly important before time-sensitive analyses, such as TRF estimation or coherence analysis. Always inspect synchronization markers, relevant delays, and alignment quality for your acquisition setup. These capabilities prepare aligned data; they do not imply that every downstream analysis is performed automatically by AlmKanal.

## Design choices and limitations

AlmKanal favors **Python-native composition**: a pipeline is an explicit sequence of Python `AlmKanalStep` objects, similar in spirit to scikit-learn pipelines. It is not intended to be API-compatible with `sklearn.pipeline.Pipeline`, nor does it require a separate YAML or JSON pipeline configuration file.

This design makes it convenient to integrate custom MNE operations and reuse only part of a workflow. For projects that need fully declarative or non-programmer-editable configurations, another pipeline architecture may be a better fit.

Some limitations to keep in mind:

- Ordering validation covers constraints explicitly declared by the processing steps; it cannot catch every scientifically inappropriate pipeline.
- Steps have different input requirements. Starting from any MNE `Raw` or `Epochs` object does not guarantee that every step can run on it.
- Processing-history-based Methods text covers recorded operations, not undocumented preprocessing performed beforehand.
- Experimental, acquisition-specific, or source-analysis functionality should be tested with representative datasets and reviewed before scientific use.

## Contributing and support

We welcome bug reports, feature requests, documentation improvements, and new preprocessing steps.

- Read the [contribution guidelines](CONTRIBUTING.md).
- Open a [GitHub issue](https://github.com/schmidtfa/AlmKanal/issues) for a bug or proposal.
- Explore the [source code](almkanal/) and [tests](tests/) for implementation examples.

If you are developing locally with Pixi, the repository provides tasks such as:

```bash
pixi install
pixi run test
pixi run lint
pixi run check_types
```

## Citation

A JOSS manuscript is being prepared. Until a formal software citation is available, please cite the [AlmKanal repository](https://github.com/schmidtfa/AlmKanal) and identify the version or commit used in your analysis.

## License and acknowledgements

AlmKanal is distributed under the [BSD 3-Clause License](LICENSE). It is developed at the Salzburg Brain Dynamics Lab, building on the scientific Python and MNE-Python ecosystems.

The name is a nod to Salzburg's *Almkanal* - and to the hope of spending less time rewriting preprocessing boilerplate and more time outside in the sun ☀️ 🍹 🏖️
