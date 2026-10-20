## Create a small recording

This tutorial uses synthetic EEG to demonstrate the API. Its parameters are examples, not a recommended preprocessing protocol. Install AlmKanal first, then run the following blocks in one Python session.

```python
from pathlib import Path

import mne
import numpy as np
from almkanal import AlmKanal, Filter, Resample, preprocessing_report

rng = np.random.default_rng(42)
sfreq = 500.0
times = np.arange(int(30 * sfreq)) / sfreq
signal = 2e-6 * np.sin(2 * np.pi * 10 * times)
data = signal[None, :] + rng.normal(0, 1e-6, (4, len(times)))
info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg")
raw = mne.io.RawArray(data, info)
raw.set_montage("standard_1020")

out = Path("almkanal-demo")
out.mkdir(exist_ok=True)
```

## Define the steps in order

```python
pipeline = AlmKanal(
    steps=[
        Filter(highpass=1.0, lowpass=40.0),
        Resample(sfreq=200),
    ]
)
```

The list is the processing order. Here, explicit low-pass filtering precedes downsampling. Constructing the pipeline checks constraints declared by its steps; data-dependent checks happen when the steps run.

## Run and inspect

```python
processed, qc_report = pipeline.run(raw.copy())

qc_report.save(
    out / "quality-control.html",
    overwrite=True,
    open_browser=False,
)
processed.save(out / "processed_raw.fif", overwrite=True)
```

`run()` returns **two objects**: the output of the final step and an MNE report. `raw.copy()` protects the original recording from steps that modify data in place. The output in this example remains continuous data.

Open the QC report and check the results. Completion without an exception is not a scientific quality check.

## Generate Methods text

```python
history_path = out / "preprocessing.json"
pipeline.generate_json(str(history_path))

methods_path = preprocessing_report(
    files=[history_path],
    out_path=out / "methods.md",
)
print(methods_path.read_text(encoding="utf-8"))
```

The Methods exporter reads the saved history and renders Jinja templates. It does not ask an AI model to describe your recording.

| Output | What to use it for |
| --- | --- |
| `processed_raw.fif` | Continue your analysis |
| `quality-control.html` | Inspect processing diagnostics |
| `preprocessing.json` | Retain the recorded processing history |
| `methods.md` | Review and edit draft Methods text |

> **Review the text.** Describe earlier processing separately, check scientific terminology and settings, and add study-specific details before using the output in a manuscript.

## Use your own recording

Replace the synthetic-data block with the appropriate MNE reader. For example:

```python
raw = mne.io.read_raw_fif("your_recording_raw.fif", preload=True)
```

Reconsider the channel types, filtering, sampling rate, and artifact handling for your study. See [working with your data](pipelines.md) and the [step catalogue](steps.md).

The complete script is available on the [examples page](examples.md).
