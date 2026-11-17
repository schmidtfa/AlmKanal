## The step contract

A step subclasses `AlmKanalStep` and implements two methods. `run(data, info)` returns a dictionary containing at least `"data"`. Other entries become processing metadata. `reports(data, report, info)` adds diagnostics to the MNE report after execution.

The `info` object carries pipeline context. Returning metadata is the route for recording a step's results; changing the copy passed to `run()` is not a way to update the pipeline's shared history.

## A continuous-data notch filter

This teaching example wraps MNE's notch filter. It intentionally accepts only `Raw` data and records the requested frequency. It is not a complete publication-grade record of every filter parameter.

```python
import mne
from attrs import define, field
from almkanal import AlmKanalStep

@define
class NotchAtFrequency(AlmKanalStep):
    frequency: float = 50.0
    must_be_before: tuple[str, ...] = field(
        default=("SourceReconstruction",), init=False
    )

    def run(self, data, info):
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("NotchAtFrequency requires continuous Raw data.")
        if not 0 < self.frequency < data.info["sfreq"] / 2:
            raise ValueError("Frequency must be between zero and Nyquist.")
        data.load_data().notch_filter(freqs=[self.frequency])
        return {
            "data": data,
            "notch_info": {"frequency_hz": self.frequency},
        }

    def reports(self, data, report, info):
        report.add_raw(
            data,
            title="After notch filtering",
            butterfly=False,
            psd=True,
        )
```

The `must_be_before` declaration makes the step participate in supported ordering checks. To make an epoched-data version, implement and test an operation appropriate to that data structure rather than simply removing the type check.

## Compose it with built-in steps

```python
from almkanal import AlmKanal, Filter, Resample

pipeline = AlmKanal([
    NotchAtFrequency(frequency=50.0),
    Filter(highpass=1.0, lowpass=40.0),
    Resample(sfreq=200),
])

# raw is your loaded continuous recording.
processed, report = pipeline.run(raw.copy())
pipeline.generate_json("preprocessing.json")
```

Choose filtering settings for your data. The sequence demonstrates composition; it is not a recommendation that this combination is necessary for every study.

## Record what matters

Prefer explicit, JSON-friendly metadata with named units. Distinguish chosen settings from data-dependent results. For a production filter step, also capture its design, bandwidth, effective length, and other scientifically relevant settings. Do not assume that the constructor's parameters are automatically recorded: include relevant values in the result returned by `run()`.

## Add Methods support

Custom-step execution and custom scientific prose are separate extension points. See [custom Methods templates](custom-reporting.md) for a `StepSpec` and Jinja partial matching `NotchAtFrequency`.

## Test the contract

Test valid and invalid inputs, ordering errors, output shape and units, metadata, and report generation. Tests should establish that invalid data fails clearly and that important settings appear in the history and rendered description.

The downloadable [custom-step example](examples.md) contains this implementation with type annotations.
