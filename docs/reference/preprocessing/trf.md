# Stimulus epoching and WAV alignment

`EpochStimuls` divides each stimulus trial into non-overlapping, fixed-length
epochs and appends one or more WAV-derived stimulus features. The resulting
epochs can be used for TRFs and other analyses.
Use `feature="envelope"` for a single feature or
`feature=["envelope", "mel_onsets"]` to append several features.
`resample_sfreq=200.0` resamples trial segments after alignment and physical-delay
correction, before computing features at 200 Hz; `None` retains the input rate.

`StimulusSpanSpec` describes the trial spans and audio files used for this workflow.

::: almkanal.almkanal_steps.stimulus_utils.EpochStimuls

::: almkanal.almkanal_steps.stimulus_utils.StimulusSpanSpec

`StimulusSpanSpec` is a supporting data class, not an `AlmKanalStep`. See the [WAV-alignment guide](../../audio-alignment.md) for workflow details.

::: almkanal.almkanal_steps.stimulus_utils.build_stimulus_epochs
