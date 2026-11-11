# Spatial filtering

`SpatialFilter` constructs spatial filters for M/EEG source analysis using MNE-Python.

Feature channels added by a preceding `StimulusEpochs` step are automatically
preserved alongside any channels listed in `chans2keep`. Their data are returned
in `extra_data` by the subsequent `SourceReconstruction` step.

::: almkanal.almkanal_steps.spatial_filter_utils.SpatialFilter
