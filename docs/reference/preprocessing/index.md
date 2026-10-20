# Preprocessing steps

The pages below document AlmKanal's public preprocessing steps directly from their Python docstrings. Each class is rendered automatically by `mkdocstrings`.

| Area | Classes |
| --- | --- |
| [Filtering and resampling](filtering.md) | `Filter`, `Resample` |
| [Maxwell filtering](maxwell.md) | `Maxwell`, `MultiBlockMaxwell` |
| [Channel cleaning and referencing](channels.md) | `EEGRANSAC`, `ReReference` |
| [Artifact and physiological signal processing](artifacts.md) | `ICA`, `PhysioCleaner` |
| [Events and epoching](epoching.md) | `Events`, `Epochs` |
| [VPixx eye-tracking alignment](eye-tracking.md) | `VPixxCleaner` |
| [TRF epoching and WAV alignment](trf.md) | `EpochTRF`, `TRFSpanSpec` |
| [Spatial filtering](spatial-filtering.md) | `SpatialFilter` |
| [Source modelling and reconstruction](source-reconstruction.md) | `ForwardModel`, `SourceReconstruction` |

A step may require continuous recordings, epoched data, particular sensor channels, or external files. Consult the relevant class documentation and the [step catalogue](../../steps.md) before using it.

For implementing new steps, see [Create a custom step](../../custom-steps.md).
