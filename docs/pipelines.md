## Start from Raw or Epochs

AlmKanal accepts MNE continuous and epoched data. A valid input object need not represent an untouched acquisition. For example, ICA or other cleaning may already have been performed outside AlmKanal.

A step still needs to support its input. Continuous-only operations do not become valid on epochs merely because the pipeline accepts both representations.

```python
# Continue an existing analysis; raw is an MNE Raw object.
from almkanal import AlmKanal, Filter, Resample

pipeline = AlmKanal([
    Filter(highpass=1.0, lowpass=40.0),
    Resample(sfreq=200),
])
processed, report = pipeline.run(raw.copy())
```

## Continue from epoched data

For epochs already low-pass filtered to 40 Hz and sampled above 200 Hz, a resampling-only pipeline can be appropriate:

```python
from almkanal import AlmKanal, Resample

# epochs already exists and has been low-pass filtered appropriately.
pipeline = AlmKanal([Resample(sfreq=200)])
processed_epochs, report = pipeline.run(epochs.copy())
```

This example does not infer or repeat earlier processing. Keep a record of those earlier steps alongside the new history. Confirm the low-pass metadata and input sampling rate before running it.

## Choose channels deliberately

`pick_params` forwards channel-type selection settings to `mne.pick_types` before the steps execute:

```python
pipeline = AlmKanal(
    steps=[Filter(highpass=1.0, lowpass=40.0)],
    pick_params={"eeg": True, "meg": False, "stim": False},
)
```

Do not discard trigger, eye-tracking, auxiliary, or recorded-audio channels before a step that needs them. For multimodal pipelines, it may be preferable to select channels within individual steps.

## Protect inputs and preserve history

Many operations act in place. Pass a copy when you need the original object later. Save the history after each completed run: the next call to `run()` clears the pipeline's earlier processing history. Save separate JSON files for separate recordings rather than treating one pipeline instance as a permanent audit log.

The JSON history is a compact processing record, **not a replacement for raw data, full numerical results, environment files, or versioned analysis code**. Its `max_elements` setting can limit saved sequence content.

## Multiple recording blocks

The pipeline also accepts homogeneous lists of `Raw` objects or homogeneous lists of `Epochs` objects. Not all steps handle lists. For example, `MultiBlockMaxwell` handles multiple raw MEG blocks and concatenates them after filtering. Use a list-aware first step and verify the representation it returns before adding ordinary single-recording steps.

## What validation does not do

AlmKanal cannot reconstruct undocumented preprocessing from your input object, prove that an analysis suits your hypothesis, or validate constraints a custom step never declared. Its role is to make the pipeline explicit and check supported rules, not to choose a study's preprocessing strategy.
