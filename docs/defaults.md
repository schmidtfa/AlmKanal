# Defaults profiles

AlmKanal uses `Defaults.generic()` automatically. A profile supplies omitted
optional arguments for pipeline steps and supported standalone functions.
Select and customize one for your script with `set_defaults()`, or temporarily with
`use_defaults()`. Explicit arguments always take precedence.

## Select and customize a profile

```python
import almkanal as ak

ak.set_defaults(
    "salzburg",
    name="our-lab",
    filter={"lowpass": 80.0},
    ica={"train": False},
)

step = ak.Filter()              # Uses lowpass=80.0
custom = ak.Filter(lowpass=30)  # Explicit arguments still win

with ak.use_defaults("generic", filter={"lowpass": 25.0}):
    temporary_step = ak.Filter()

assert ak.get_defaults().name == "our-lab"
ak.set_defaults()  # Reset to the general profile.
```

`set_defaults()` and `use_defaults()` accept `"generic"`, `"salzburg"`, or a
`Defaults` object. Both accept the same section overrides and optional `name`
and `version`. `set_defaults()` returns the activated profile, and
`with use_defaults(...) as profile:` exposes the temporary profile.

Each call starts from its selected profile, which is `"generic"` when omitted.
It does not accumulate settings from earlier calls. To extend the active
settings explicitly, use `ak.set_defaults(ak.get_defaults(), filter={"lowpass": 60})`.
Invalid profiles or settings raise an error before changing the active defaults.

For reusable lab profiles, construct a `Defaults` object separately. The
lower-level `configure(profile)` activates an already-built object:

```python
from almkanal import AlmKanal, Defaults, Filter, ICA, configure, get_defaults

lab = Defaults.generic().with_overrides(
    name="our-lab",
    version="1",
    filter={"highpass": 0.5, "lowpass": 80.0},
    ica={"random_state": 123, "train": False},
    events={"stim_channel": "TRIGGER"},
)
configure(lab)

pipeline = AlmKanal(steps=[Filter(), ICA(ecg=False)])
assert get_defaults().name == "our-lab"
assert pipeline.steps[0].lowpass == 80.0
```

These examples construct pipelines without loading data. Run a constructed
pipeline using `processed_data, report = pipeline.run(raw)`, where `raw` is
your loaded MNE recording.

`Defaults()` and `Defaults.generic()` are equivalent. Use
`configure(Defaults.generic())` to reset the active profile. Your profile can be
defined in a shared Python module and imported by your lab's scripts. Loading
profiles from TOML, YAML, or JSON files is not implemented.

`with_overrides()` returns a new profile and leaves its source unchanged.
Unspecified sections and options retain their previous values. Unknown section
or option names and incompatible value types raise errors immediately.
Scientific validity, such as whether a cutoff suits a recording's sampling
rate, remains the responsibility of the processing step.

A dictionary supplied as an option replaces that entire option; it is not
deep-merged. For example, supply a complete IIR filter configuration:

```python
lab = lab.with_overrides(
    filter={"method": "iir", "iir_params": {"order": 4, "ftype": "butter"}},
)
```

Call `configure(lab)` again to activate this new profile. Give shared profiles
a descriptive name and update their version when changing agreed settings.

## Temporarily select defaults

```python
from almkanal import Defaults, Filter, configure, get_defaults, use_defaults

configure(Defaults.generic())

with use_defaults(Defaults.salzburg()):
    salzburg_filter = Filter()
    with use_defaults(Defaults.generic().with_overrides(filter={"lowpass": 25.0})):
        comparison_filter = Filter()
    assert get_defaults().name == "salzburg"

assert get_defaults().name == "generic"
assert comparison_filter.lowpass == 25.0
```

Contexts can be nested and restore the previous profile even if an exception
leaves the block. Steps built inside the block can be run outside it.

## When values are resolved

Pipeline steps capture their profile and omitted defaults **when constructed**.
Configure the profile before writing `Filter()`, `ICA()`, or other step
constructors. Changing the active profile later does not change existing steps.
Built-in steps run their internal helpers and callbacks under their captured
profile, so a delayed callback sees the same defaults as its owning step. An
explicit `EpochTRF(fallback_drift_us_per_s=...)` also becomes the default for
trial-end inference inside its span callback. An explicit argument inside the
callback still takes precedence.

Standalone functions resolve their omitted arguments when called. Explicit
positional and keyword arguments take precedence, including `None`, `False`,
and `0` wherever those values are supported:

```python
from almkanal import Defaults, Filter, ICA, use_defaults

with use_defaults(Defaults.salzburg()):
    unfiltered_lowpass = Filter(lowpass=None)
    without_train_removal = ICA(train=False)
```

Required inputs remain required. For example, `Resample(sfreq=200)` still needs
`sfreq`, `ForwardModel` still needs `subject_id` and `subjects_dir`, and
`EpochTRF` still needs its span callback and audio path. Profiles do not supply
recordings or replace the pipeline's step ordering requirements.

## Supported sections

Use section names as keyword arguments to `set_defaults()`, `use_defaults()`,
or `with_overrides()`. The examples below
are selected settings; `Defaults.generic().section("filter")` returns all
settings in that section. The typed definitions in
[`almkanal/_defaults_schema.py`](https://github.com/schmidtfa/AlmKanal/blob/feat/configurable-defaults/almkanal/_defaults_schema.py)
list every option.

| Section | Used for | Example settings |
| --- | --- | --- |
| `filter` | `Filter` | `highpass`, `lowpass`, `method`, `iir_params` |
| `resample` | `Resample` | `npad`, `window`, `method` |
| `ica` | `ICA`, `run_ica` | `method`, `resample_freq`, `eog`, `ecg`, `train` |
| `events` | `Events` | `stim_channel`, `shortest_event`, `mask` |
| `physio` | `PhysioCleaner`, `run_bio_preproc` | `ecg`, `resp`, `eog`, `emg` |
| `maxwell` | Maxwell processing | `mw_coord_frame`, `mw_calibration_file`, `mw_st_duration` |
| `ransac` | `EEGRANSAC` | `ransac_epoch_duration`, `min_corr`, `n_jobs` |
| `rereference` | `ReReference` | `ref_channels`, `projection`, `ch_type` |
| `epochs` | `Epochs` | `tmin`, `tmax`, `baseline`, `reject` |
| `forward_model` | `ForwardModel`, `make_fwd` | `source`, `spacing`, `bem_conductivity` |
| `spatial_filter` | `SpatialFilter`, `comp_spatial_filters` | `lcmv_reg`, `lcmv_pick_ori`, `lcmv_weight_norm` |
| `source_reconstruction` | `SourceReconstruction`, parcellation | `return_parc`, `atlas`, `label_mode` |
| `trf` | `EpochTRF`, TRF epoch helpers | `hw_delay_s`, `epoch_len_s`, `fallback_drift_us_per_s` |
| `pipeline` | `AlmKanal` | `pick_params` |
| `vpixx` | `VPixxCleaner`, `VPixxConfig`, eye loading/alignment helpers | `screen_resolution`, `screen_size`, `screen_distance`, `meg_stim_channel`, `concatenate_raws` |
| `ica_eog` | `eog_ica_from_meg` | `left_eog_chs`, `right_eog_chs`, `threshold`, `tol` |
| `ica_train` | `find_train_ica` | `duration`, `overlap`, `hmax` |
| `trials` | Trial span helpers | `end_triggers`, `infer_missing_ends`, `base_audio_path` |
| `alignment` | Raw/WAV alignment | `sync_sfreq`, `max_lag_s`, `min_corr` |
| `audio` | Audio feature extraction | `feature`, `target_fs`, `cutoff_hz`, `n_mels` |

Related helpers share settings where appropriate. In particular, trial span
inference and TRF processing use `trf.fallback_drift_us_per_s`, and
`find_train_ica` uses `ica.train_thresh` for its `peak_threshold` argument.

## General defaults and migration from Salzburg defaults

The general profile removes these historical lab assumptions:

| Setting | General profile | Salzburg profile |
| --- | --- | --- |
| `ica.train` | `False` | `True` |
| `trf.hw_delay_s` | `0.0` seconds | `0.0165` seconds |
| `trf.fallback_drift_us_per_s` | `0.0` | `499.0` |
| `trf.audio_channels` | `None` | `("MISC007", "MISC008")` |
| `vpixx.meg_stim_channel` | `None` (MNE detection) | `"STI101"` |
| `vpixx.meg_min_duration` | `0.0` seconds | `0.002` seconds |
| `vpixx.meg_max_trigger` | `None` (keep all codes) | `4096` (exclude this code and above) |
| `vpixx.screen_distance` | Must be configured for radians | `0.82` m |
| `vpixx.screen_size` | Must be configured for radians | `(0.61, 0.34)` m |
| `vpixx.screen_resolution` | Must be configured for radians | `(1920, 1080)` pixels |
| `ica_eog.left_eog_chs` | Must be configured when using the helper | `["MEG0121", "MEG0311"]` |
| `ica_eog.right_eog_chs` | Must be configured when using the helper | `["MEG1211", "MEG1411"]` |

To retain those assumptions in an existing Salzburg script, add
`set_defaults("salzburg")` before constructing its steps or calling its
helpers. General defaults do not establish the correct hardware calibration for
another lab; configure measured values for your setup.

The Salzburg profile enables recorded-audio alignment using `MISC007` and
`MISC008`. Pass `audio_channels=None` to `EpochTRF` or `build_trf_epochs` when
recorded audio is unavailable, or override the profile with
`set_defaults("salzburg", trf={"audio_channels": None})`.

## VPixx eye tracking

The new `VPixxCleaner` step and standalone eye-tracking functions share the
`vpixx` section. The sampling rate is read from the recording's timestamps.
Configure screen geometry in metres and pixels before constructing steps:

```python
import almkanal as ak

ak.set_defaults(
    name="our-eye-setup",
    vpixx={
        "screen_resolution": (1920, 1080),
        "screen_size": (0.53, 0.30),
        "screen_distance": 0.65,
        "meg_stim_channel": "TRIGGER",
        "blink_buffer_vpixx": (0.05, 0.2),
        "concatenate_raws": True,
    },
)

step = ak.VPixxCleaner(eye_path="eye.mat", interpolate_blinks=False)
```

The numbers here illustrate the API; replace them with your equipment's values.
`VPixxCleaner` captures the profile when constructed, retains it for each block,
and records its resolved parameters in the pipeline JSON. `eye_path` remains a
required input. `concatenate_raws` applies to list input; a single recording
still returns a single Raw object.
The `align` setting controls the standalone `add_eye_tracking_data()` helper;
`VPixxCleaner` always aligns eye tracking to its input recordings.

`VPixxConfig()` also captures omitted values at construction. The standalone
`load_eyetracking_data()`, `add_eye_tracking_data()`, and `align_eye_to_meg()`
resolve omitted options when called. An omitted or `None` `config` builds a
fresh `VPixxConfig` from the active profile, replacing the former configuration
object fixed at import time. An explicit config supplies its own geometry and
calibration settings; loading and alignment options still use their corresponding
defaults unless passed explicitly.

`vpixx.blink_buffer_vpixx` supplies `VPixxConfig.blink_buffer`.
`vpixx.blink_buffer` is the loading/interpolation override: `None` uses the
config's buffer, and an explicit `(0.0, 0.0)` uses no padding. Explicit `None`
also overrides a profile's non-None loading buffer.

With the generic profile, conversion to radians raises a clear error until
`screen_resolution`, `screen_size`, and `screen_distance` are supplied. For
pixel-only processing, use `convert_to_radians=False`. Explicitly setting one
geometry value to `None` does not silently restore the Salzburg value.
`meg_max_trigger=None` retains all trigger codes, and `meg_stim_channel=None`
uses MNE's stimulus-channel detection.

### Migration from the previous eye helper

Upstream replaced `clean_pixx_eye_data()` with the VPixx implementation. The old
`eye` profile section is replaced by `vpixx`; old names now raise an unknown-section
error. Replace `distance` in centimetres with `screen_distance` in metres,
`screen_rect` with the two-value `screen_resolution`, and `screen_width` with
both physical dimensions in `screen_size`. There is no `tpixx_fs` setting:
timestamps determine the sampling rate. The Salzburg profile follows the new
VPixx setup on main (61 × 34 cm screen, 82 cm distance); the former helper used
a 63 cm screen width. The old `gaze_lims`, `filter_settings`, and `annotate_bads`
options have no direct equivalents in the new loader.

Surrogate EOG extraction likewise needs channel names appropriate to the
recording. Configure `ica_eog` when using `eog_ica_from_meg`, or explicitly supply
its channel arguments.

The standalone `run_ica()` now shares the step's `ica.resample_freq` default of
200 Hz. It previously defaulted to `None`. Pass `resample_freq=None` directly to
that helper to preserve its previous no-resampling behavior.

## Inspecting and recording settings

Profiles are immutable through their public API and protect nested values by
copying them. `profile.filter` returns a typed section copy;
`profile.section("filter")` returns a detached dictionary; `profile.to_dict()`
returns a detached snapshot of all sections plus the profile name and version.
Changing a returned dictionary does not change the active profile. Use
`with_overrides()` for customization.

After a pipeline runs, each built-in step's result includes `defaults` metadata
with `profile`, `version`, and resolved `parameters` for its configurable fields.
This metadata is included by `pipeline.generate_json()` alongside existing step
information. Explicit overrides are reflected in the resolved parameters.
Profile names alone are not a substitute for saving those settings and the
profile definition, particularly when callbacks use additional helper defaults.

The JSON output contains an ordered `processing_history` list. Each entry has
`step` and `info` fields, with defaults metadata inside `info`. Repeated steps
retain their individual settings and results whenever the step allows repeated
use.

## Concurrent execution

The active profile is stored in a Python `ContextVar`. Async tasks inherit the
context in which they are created; subsequent changes remain local to each
task. New threads do not inherit the profile unless their context is explicitly
copied. Configure worker processes explicitly or send already-constructed,
serialized steps, which retain their captured profiles. Do not rely on one
process calling `configure()` to configure all workers.
