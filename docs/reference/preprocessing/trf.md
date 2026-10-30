# TRF epoching and WAV alignment

`EpochTRF` prepares epochs with one or more WAV-derived stimulus features.
Use `feature="envelope"` for a single feature or
`feature=["envelope", "mel_onsets"]` to append several features.
`resample_sfreq=200.0` resamples trial segments after alignment and physical-delay
correction, before computing features at 200 Hz; `None` retains the input rate. `TRFSpanSpec` describes the trial spans and audio files used for this workflow.

::: almkanal.almkanal_steps.trf_utils.EpochTRF

::: almkanal.almkanal_steps.trf_utils.TRFSpanSpec

`TRFSpanSpec` is a supporting data class, not an `AlmKanalStep`. See the [WAV-alignment guide](../../audio-alignment.md) for workflow details.
