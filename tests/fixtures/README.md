`legacy_epoch_trf.pickle` and `legacy_epoch_trf_defaults.pickle` were generated
with pickle protocol 0 before the EpochStimuls rename, from commit `6ae1cd1`.
They contain synthetic settings only. The step uses `dict` as a pickleable
callback, the Salzburg profile, and `resample_sfreq=125.0` with envelope and
rectify features. They verify that actual historical class paths still load.


`legacy_trf_span_spec.pickle` was captured before the StimulusSpanSpec rename.
It contains a synthetic `story` span, speech-condition metadata, and a relative
WAV filename. It checks that span specifications saved under the historical
TRFSpanSpec class path retain their fields when loaded.
