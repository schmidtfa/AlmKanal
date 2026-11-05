## Define trials and their WAV files

`StimulusSpanSpec` describes raw-sample trial spans and their corresponding stimuli. `EpochStimuls` uses a function that creates this specification from the input recording, prepares WAV features, and returns fixed-length epochs.

The following is an acquisition-dependent example. It assumes known onset/end trigger codes and a recorded-audio channel. Replace those values with your experiment's mapping.

```python
from pathlib import Path

import mne
from almkanal import AlmKanal, EpochStimuls, StimulusSpanSpec

AUDIO_DIR = Path("stimuli")

def make_spans(raw):
    return StimulusSpanSpec.from_events(
        raw,
        onset_trigger_to_wav={1: "story.wav"},
        end_triggers=[2],
        stim_channel="STI101",
        base_audio_path=AUDIO_DIR,
    )

raw = mne.io.read_raw_fif("your_recording_raw.fif", preload=True)
pipeline = AlmKanal([
    EpochStimuls(
        gen_span_spec=make_spans,
        base_audio_path=AUDIO_DIR,
        audio_channels=["AUDIO"],  # Your actual recorded-audio channel.
        feature=["envelope", "pitch_voicing"],
        resample_sfreq=200.0,
        epoch_len_s=5.0,
        hw_delay_s=0.0,  # Only when the reference represents sound at the ears.
        on_alignment_error="raise",
    ),
])

epochs, qc_report = pipeline.run(raw.copy())
pipeline.generate_json("audio_preprocessing.json")
```

This prepares data for downstream analysis. It does not, by itself, fit a TRF model or compute coherence.

## Distinguish three alignment modes

| Configuration | What happens |
| --- | --- |
| Recorded `audio_channels` supplied | Offset and clock drift are estimated from recorded audio and the WAV |
| No audio channels, `realign_without_audio=True` | Alignment uses WAV duration and an assumed drift; onset is assumed, not measured |
| No audio channels, `realign_without_audio=False` | No clock realignment is performed; trial timing and any physical-delay correction are used |

Do not describe an assumed-drift correction as an empirical synchronization estimate. Recorded audio takes precedence when supplied; failed audio fits do not silently fall back to assumed drift.

## Measure physical delays for your setup

Clock alignment and hardware-delay correction solve different problems. `hw_delay_s` is applied after realignment. Positive values advance neural events relative to the WAV features; the feature channels are not shifted.

The package's default delay and fallback drift reflect particular assumptions. They are not universal constants. Use a measured delay for your acquisition system, or zero when your recorded reference already represents sound arrival at the ears.

## Inspect alignment diagnostics

Review the QC output and recorded metadata for offset, drift, residual error, correspondence strength, rejected or skipped trials, and retained epochs. Long stimuli can need a wider alignment search interval to accommodate accumulated clock drift.

Quality warnings can be emitted even when an alignment is applied. `on_alignment_error="raise"` stops actual failures; it is not an automatic rejection rule for every questionable fit. Use study-specific acceptance criteria and keep them with the analysis.

Preserve trigger and recorded-audio channels until they have been used, and avoid preprocessing away the bandwidth needed for alignment.

## Choose the sampling frequency and audio features

`resample_sfreq` sets the output sampling frequency in Hz. With `None`, the
input sampling frequency is retained. Trial spans and trigger samples always
refer to the original recording. Alignment and the sample-rounded physical-delay
correction run at that original rate. Each continuous trial segment is then
resampled before audio features are computed at the output rate and appended.
This preserves the available recorded-audio bandwidth for alignment and avoids
resampling already appended features. An earlier resampling step reduces the
bandwidth available to alignment.

Segment resampling uses MNE's FFT method with a boxcar frequency-domain window,
automatic padding, and `reflect_limited` edge padding. MNE applies anti-aliasing
internally. Annotations are preserved when `preserve_annotations=True`; epochs
overlapping bad annotations are rejected. Reports record the original and output
sampling frequencies, resampling method, and processing order.

`feature` accepts either one feature name or a nonempty sequence of distinct
feature names. Channels are appended in the requested feature order as `misc`
channels. Single-channel features use the requested feature name. Multichannel
features use that name followed by component or band identifiers:

| Feature | Appended channels |
| --- | --- |
| `envelope` | `envelope` (RMS amplitude) |
| `rectify` | `rectify_pos_rect`, `rectify_neg_rect` |
| `mel` | `mel_1_<frequency>Hz`, … |
| `flux` | `flux` (onset strength) |
| `mel_onsets` | `mel_onsets_1_<frequency>Hz`, … |
| `pitch` | `pitch` (fundamental frequency in Hz) |
| `pitch_voicing` | `pitch_voicing_log2_f0_centered`, `pitch_voicing_voicing` |

Mel band indices start at one and frequency labels are rounded band centers.
Feature names must not repeat, and appended channel names must not overlap
recording channels. `audio_cutoff_hz` controls the feature low-pass filter;
feature extraction details are described by `prepare_audio`. Features share the
output sampling grid; only the common recording/feature duration contributes
epochs, and incomplete trailing epochs are discarded.

The standalone `build_stimulus_epochs` helper accepts the same `feature` and
`resample_sfreq` parameters with an existing `StimulusSpanSpec`.

## Share stimulus-processing defaults

Use the `stimuli` defaults section to configure shared settings for
`EpochStimuls`, `build_stimulus_epochs`, trial-end inference, and assumed drift:

```python
from almkanal import Defaults, set_defaults

profile = Defaults.generic().with_overrides(
    stimuli={
        "resample_sfreq": 200.0,
        "feature": ["envelope", "pitch_voicing"],
        "epoch_len_s": 5.0,
    },
)
set_defaults(profile)

# Inspect a detached copy of the stimulus settings.
settings = profile.stimuli
```

Explicit step or helper arguments take precedence over profile settings. Each
step captures its profile when constructed; changing the active defaults later
does not change an existing step's configuration.
