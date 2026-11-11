# Stimulus epoching and WAV alignment

`StimulusEpochs` divides each stimulus trial into non-overlapping, fixed-length
epochs and appends one or more WAV-derived stimulus features. The resulting
epochs can be used for TRFs and other analyses.
Use `feature="envelope"` for a single feature or
`feature=["envelope", "mel_onsets"]` to append several features.
All generated feature channels are automatically preserved by a later
`SpatialFilter` and returned alongside source estimates in
`SourceReconstruction`'s `extra_data` dictionary. There is no need to repeat
their names in `SpatialFilter.chans2keep`; that list can specify additional
recording channels to retain.
`resample_sfreq=200.0` resamples trial segments after alignment and physical-delay
correction, before computing features at 200 Hz; `None` retains the input rate.
Changing the sampling rate with a later `Resample` step emits a warning:
resampling appended features can distort binary (0/1) values, introducing
intermediate values or ringing. Use `StimulusEpochs(resample_sfreq=...)` to
compute features at the desired output rate instead.

`StimulusSpanSpec` describes the trial spans and audio files used for this workflow.

::: almkanal.almkanal_steps.stimulus_utils.StimulusEpochs

::: almkanal.almkanal_steps.stimulus_utils.StimulusSpanSpec

`StimulusSpanSpec` is a supporting data class, not an `AlmKanalStep`. See the [WAV-alignment guide](../../audio-alignment.md) for workflow details.

::: almkanal.almkanal_steps.stimulus_utils.build_stimulus_epochs
