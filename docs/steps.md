## Step catalogue

These names are exported by the inspected `almkanal` package. This catalogue is a navigation guide, not a recommendation to apply every operation.

| Area | Components | Role |
| --- | --- | --- |
| Temporal processing | `Filter`, `Resample` | Filtering and changes in sampling frequency |
| MEG channels | `Maxwell`, `MultiBlockMaxwell` | Maxwell filtering for a recording or multiple blocks |
| EEG channels | `ReReference`, `EEGRANSAC` | Referencing and RANSAC-based channel handling |
| Artifact and physiological processing | `ICA`, `PhysioCleaner` | Independent-component and physiological-signal workflows |
| Events | `Events`, `Epochs` | Trigger extraction and epoch creation |
| Eye tracking | `VPixxCleaner` | Load, clean, and align supported VPixx recordings |
| Stimulus epochs | `EpochStimuls`, `StimulusSpanSpec` | Trial specifications, WAV features, and aligned epochs |
| Source analysis | `ForwardModel`, `SpatialFilter`, `SourceReconstruction` | Building blocks for source-space workflows |
| Pipeline and output | `AlmKanal`, `AlmKanalStep`, `preprocessing_report` | Composition, extension, and scientific reporting |

## Use the right input

`Filter` and `Resample` target continuous or epoched data. Maxwell filtering, event detection, and audio-alignment workflows require continuous recordings. `MultiBlockMaxwell` takes raw recording blocks. Source-analysis steps additionally depend on study-specific anatomical and recording information.

Inspect each class's current implementation and tests before building a hardware-specific workflow. A class being available at the top level does not imply that it supports every input representation or has no external requirements.

## Explore full workflows

The repository's [template pipelines](https://github.com/schmidtfa/AlmKanal/tree/main/template_pipelines) provide examples of more extensive workflows. Treat them as study-dependent starting points rather than universal defaults.

For details on individual processing steps, see the [API reference](reference/preprocessing/index.md). More specialized workflows are covered in the guides on [eye-tracking alignment](eye-tracking.md), [audio alignment](audio-alignment.md), and [creating custom steps](custom-steps.md) when you need an operation not included here.
