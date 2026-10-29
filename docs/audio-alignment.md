## Define trials and their WAV files

`TRFSpanSpec` describes raw-sample trial spans and their corresponding stimuli. `EpochTRF` uses a function that creates this specification from the input recording, prepares WAV features, and returns fixed-length epochs.

The following is an acquisition-dependent example. It assumes known onset/end trigger codes and a recorded-audio channel. Replace those values with your experiment's mapping.

```python
from pathlib import Path

import mne
from almkanal import AlmKanal, EpochTRF, TRFSpanSpec

AUDIO_DIR = Path("stimuli")

def make_spans(raw):
    return TRFSpanSpec.from_events(
        raw,
        onset_trigger_to_wav={1: "story.wav"},
        end_triggers=[2],
        stim_channel="STI101",
        base_audio_path=AUDIO_DIR,
    )

raw = mne.io.read_raw_fif("your_recording_raw.fif", preload=True)
pipeline = AlmKanal([
    EpochTRF(
        gen_span_spec=make_spans,
        base_audio_path=AUDIO_DIR,
        audio_channels=["AUDIO"],  # Your actual recorded-audio channel.
        feature="envelope",
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
