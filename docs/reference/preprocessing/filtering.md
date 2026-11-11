# Filtering and resampling

Temporal filtering and changes in sampling frequency for continuous or epoched M/EEG data.

Use `StimulusEpochs(resample_sfreq=...)` to change the sampling rate before
stimulus features are computed. A later `Resample` step warns when it changes
the rate because resampling appended features can distort binary (0/1) values.

::: almkanal.almkanal_steps.filter_utils.Filter

::: almkanal.almkanal_steps.filter_utils.Resample
