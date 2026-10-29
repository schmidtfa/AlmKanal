# Defaults for your lab or study

AlmKanal lets you choose shared defaults for your whole lab or for a particular
study. Values you supply in individual function calls still take precedence.

## Use a lab profile

To use the built-in Salzburg settings, add this near the beginning of your script:

```python
import almkanal as ak

ak.set_defaults("salzburg")
```

Then build your pipeline as usual:

```python
pipeline = ak.AlmKanal(
    steps=[
        ak.Filter(),
        ak.Resample(sfreq=200),
    ]
)
```

The steps use the selected defaults for any optional arguments you leave out.
Once you have loaded your MNE recording as `raw`, run the pipeline with
`processed_raw, report = pipeline.run(raw)`.

If you do not select a profile, AlmKanal uses general defaults automatically.
Calling `ak.set_defaults()` resets the active settings to those general defaults.

## Adjust settings for a study

You can start from a lab profile and change just the settings needed for a study:

```python
ak.set_defaults("salzburg", filter={"lowpass": 30.0})

step = ak.Filter()              # Uses the study's 30 Hz low-pass setting.
custom = ak.Filter(lowpass=20)  # Uses 20 Hz for this step only.
```

Options are grouped into sections such as `filter`, `ica`, `trf`, and `vpixx`.
Unspecified options retain their values from the selected profile. The complete
list of sections, types, and general defaults is at the end of this guide.

## Create and save your own lab profile

Another lab can define a reusable profile using `Defaults.generic()` as a
starting point. Save the following in a file called `lab_defaults.py`, alongside
your analysis scripts:

```python
# lab_defaults.py
from almkanal import Defaults

LAB_DEFAULTS = Defaults.generic().with_overrides(
    name="our-lab",
    version="1",
    filter={"highpass": 0.5, "lowpass": 40.0},
    events={"stim_channel": "TRIGGER"},
)
```

Replace these example values with your lab's settings. `with_overrides()` creates
a new profile and leaves the original unchanged. Defining a profile does not
activate it.

### Load and use the saved profile

Import the profile in any analysis script and activate it with `set_defaults()`:

```python
import almkanal as ak
from lab_defaults import LAB_DEFAULTS

ak.set_defaults(LAB_DEFAULTS)

pipeline = ak.AlmKanal(
    steps=[
        ak.Filter(),
        ak.Resample(sfreq=200),
    ]
)
```

Importing `lab_defaults.py` recreates the profile from the saved values. To share
it across projects, put the module in a shared Python package that your scripts
can import. Keeping this file under version control also records changes to your
lab's settings. Update its `version` when changing an agreed configuration.

The profile's `name` is a label for processing records; it does not register a
new string shortcut. Use `ak.set_defaults(LAB_DEFAULTS)` to load your own profile.
The built-in string shortcuts are `"generic"` and `"salzburg"`.

Profiles are saved here as ordinary Python files. Built-in loaders for JSON,
YAML, or TOML profile files are not currently available.

### Derive a study profile from your lab profile

A study can reuse the lab settings while overriding a few values:

```python
# study_defaults.py
from lab_defaults import LAB_DEFAULTS

STUDY_DEFAULTS = LAB_DEFAULTS.with_overrides(
    name="auditory-study",
    version="1",
    filter={"lowpass": 30.0},
)
```

The lab profile stays unchanged. Save this definition in `study_defaults.py`,
then load it in your analysis script:

```python
import almkanal as ak
from study_defaults import STUDY_DEFAULTS

ak.set_defaults(STUDY_DEFAULTS)
```

## General and Salzburg settings

The Salzburg profile starts from the general defaults and changes these values:

| Setting | General profile | Salzburg profile |
| --- | --- | --- |
| `ica.train` | `False` | `True` |
| `trf.hw_delay_s` | `0.0` seconds | `0.0165` seconds |
| `trf.fallback_drift_us_per_s` | `0.0` µs/s | `499.0` µs/s |
| `trf.audio_channels` | `None` | `("MISC007", "MISC008")` |
| `vpixx.meg_stim_channel` | `None` (MNE detection) | `"STI101"` |
| `vpixx.meg_min_duration` | `0.0` seconds | `0.002` seconds |
| `vpixx.meg_max_trigger` | `None` (keep all codes) | `4096` (exclude this code and above) |
| `vpixx.screen_distance` | Must be configured for radians | `0.82` m |
| `vpixx.screen_size` | Must be configured for radians | `(0.61, 0.34)` m |
| `vpixx.screen_resolution` | Must be configured for radians | `(1920, 1080)` pixels |
| `ica_eog.left_eog_chs` | Must be configured when using the helper | `["MEG0121", "MEG0311"]` |
| `ica_eog.right_eog_chs` | Must be configured when using the helper | `["MEG1211", "MEG1411"]` |

For another setup, supply its measured geometry, timing values, and channel
names. Selecting a profile does not check whether its hardware settings match
your recording.

Salzburg's audio channels enable recorded-audio alignment during TRF processing.
If a recording has no audio channels, use
`ak.set_defaults("salzburg", trf={"audio_channels": None})` or pass
`audio_channels=None` to the individual `EpochTRF` step.

## When settings take effect

Set defaults **before constructing your steps**. Existing steps keep the values
and profile they captured when they were created:

```python
ak.set_defaults("generic", filter={"lowpass": 30.0})
first = ak.Filter()

ak.set_defaults("generic", filter={"lowpass": 40.0})
second = ak.Filter()

print(first.lowpass)   # 30.0
print(second.lowpass)  # 40.0
```

Built-in steps also run their internal helpers and callbacks under their
captured profile. Standalone functions resolve omitted arguments when called.

Explicit arguments always win, including `None`, `False`, and `0` where those
values are supported:

```python
ak.set_defaults("salzburg")

unfiltered_lowpass = ak.Filter(lowpass=None)
without_train_removal = ak.ICA(train=False)
```

Required inputs remain required. For example, `Resample` still needs `sfreq`,
`ForwardModel` needs `subject_id` and `subjects_dir`, and `EpochTRF` needs its
span callback and audio path. Profiles do not replace the pipeline's step
ordering requirements.

### Repeated calls and validation

Each `set_defaults()` call starts from the selected profile, which is
`"generic"` when omitted. It does not accumulate earlier customizations. To
extend your active settings explicitly, pass `get_defaults()`:

```python
ak.set_defaults("salzburg", filter={"lowpass": 30.0})
ak.set_defaults(ak.get_defaults(), ica={"random_state": 123})
```

This keeps the 30 Hz low-pass setting and changes the ICA random seed.
`set_defaults()` returns the activated profile if you want to keep it in a
variable.

Unknown sections or options and incompatible value types raise an error before
the active defaults change. Processing functions still check scientific
constraints, such as whether a cutoff suits the recording's sampling rate.

A dictionary supplied as an option replaces that whole option; dictionaries
are not recursively merged. For example, supply a complete IIR configuration:

```python
ak.set_defaults(
    LAB_DEFAULTS,
    filter={"method": "iir", "iir_params": {"order": 4, "ftype": "butter"}},
)
```

The lower-level `configure(profile)` activates an already-built `Defaults`
object. For most scripts, `set_defaults()` is sufficient.

## VPixx eye tracking

`VPixxCleaner` and the standalone eye-tracking helpers share the `vpixx`
section. Sampling rate comes from the recording's timestamps. Screen size and
viewing distance are in metres; resolution is in pixels:

```python
ak.set_defaults(
    LAB_DEFAULTS,
    vpixx={
        "screen_resolution": (1920, 1080),
        "screen_size": (0.53, 0.30),
        "screen_distance": 0.65,
        "meg_stim_channel": "TRIGGER",
    },
)

step = ak.VPixxCleaner(eye_path="eye.mat", interpolate_blinks=False)
```

Replace the example geometry with your equipment's values. The general profile
requires these three geometry settings for conversion to radians. Pixel-only
processing with `convert_to_radians=False` does not require geometry.

`eye_path` remains required. Set `concatenate_raws=True` to combine a list of
recording blocks after cleaning; a single recording returns a single Raw object.
The `align` option controls only the standalone `add_eye_tracking_data()` helper;
`VPixxCleaner` always aligns eye tracking to its input recordings.

`VPixxConfig()` captures its omitted values at construction. The standalone
`load_eyetracking_data()`, `add_eye_tracking_data()`, and `align_eye_to_meg()`
resolve omitted options at call time. An omitted or `None` `config` builds a
fresh `VPixxConfig` from the active profile. An explicit config supplies its own
geometry and calibration settings; other loading and alignment arguments still
use the active profile unless explicitly supplied.

`vpixx.blink_buffer_vpixx` supplies `VPixxConfig.blink_buffer`.
`vpixx.blink_buffer` overrides that buffer during loading; `None` uses the
config's buffer and `(0.0, 0.0)` uses no padding. Explicit `None` also overrides
a profile's non-None loading buffer.

`meg_max_trigger=None` retains all trigger codes. `meg_stim_channel=None` uses
MNE's stimulus-channel detection.

## Temporary settings

For a temporary comparison, `use_defaults()` accepts the same profile and
section overrides as `set_defaults()`:

```python
ak.set_defaults("salzburg")

with ak.use_defaults("generic", filter={"lowpass": 25.0}):
    comparison_filter = ak.Filter()

# Salzburg is active again; comparison_filter keeps its 25 Hz setting.
```

Blocks can be nested. The previous profile is restored even if an exception
leaves the block. Steps created inside can be run afterward.

## Inspect and record settings

```python
profile = ak.get_defaults()

print(profile.name)
print(profile.filter.lowpass)
filter_settings = profile.section("filter")
all_settings = profile.to_dict()
```

Profiles protect their values from accidental changes. Section objects and
returned dictionaries are independent copies; editing them does not change the
active profile. Use `with_overrides()` or `set_defaults()` to customize it.

After a pipeline runs, each built-in step records its profile name, version,
and resolved configurable parameters, including explicit overrides. Save the
processing history with:

```python
pipeline.generate_json("preprocessing.json")
```

The JSON contains an ordered `processing_history` list, with separate settings
for each occurrence of a step. It is a processing record, not a loadable lab
profile. Keep your profile's Python file alongside the analysis, particularly
when callbacks also use helper defaults.

## Concurrent execution

Active settings are local to the current execution context. Async tasks inherit
the context in which they are created; later changes remain local to each task.
Configure worker threads and processes explicitly, or pass an appropriate
context or already-constructed steps. Setting a profile in one process does not
configure every worker.

## Complete settings reference

Use these section names as keyword arguments to `set_defaults()`,
`use_defaults()`, or `with_overrides()`. Every configurable option is listed below
with its accepted type and **general-profile default**. The Salzburg differences
are listed earlier in this guide.

`None` is an explicit value, not an instruction to fall back to another profile.
`Literal[...]` lists allowed choices; `Sequence` accepts lists or tuples and
`Mapping` accepts dictionaries. `Any` means that the profile does not restrict
the value's type; the processing function determines which inputs it supports.

Related helpers share settings where appropriate. Trial span inference uses
`trf.fallback_drift_us_per_s`, and `find_train_ica` uses `ica.train_thresh` for its
`peak_threshold` argument. An explicit `EpochTRF(fallback_drift_us_per_s=...)`
also supplies the default inside its span callback; an explicit callback
argument still takes precedence.

### `filter`

Used by `Filter`.

| Option | Type | General default |
| --- | --- | --- |
| `highpass` | <code>float &#124; None</code> | `0.1` |
| `lowpass` | <code>float &#124; None</code> | `40.0` |
| `picks` | <code>Any</code> | `None` |
| `filter_length` | <code>str &#124; int</code> | `'auto'` |
| `l_trans_bandwidth` | <code>float &#124; Literal['auto']</code> | `'auto'` |
| `h_trans_bandwidth` | <code>float &#124; Literal['auto']</code> | `'auto'` |
| `n_jobs` | <code>int &#124; str &#124; None</code> | `None` |
| `method` | <code>str</code> | `'fir'` |
| `iir_params` | <code>dict[str, Any] &#124; None</code> | `None` |
| `phase` | <code>str</code> | `'zero'` |
| `fir_window` | <code>str</code> | `'hamming'` |
| `fir_design` | <code>str</code> | `'firwin'` |
| `skip_by_annotation` | <code>str &#124; tuple[str, ...] &#124; list[str]</code> | `('edge', 'bad_acq_skip')` |
| `pad` | <code>str</code> | `'reflect_limited'` |

### `resample`

Used by `Resample` (the target `sfreq` is required separately).

| Option | Type | General default |
| --- | --- | --- |
| `npad` | <code>str</code> | `'auto'` |
| `window` | <code>str &#124; tuple[str, float]</code> | `'auto'` |
| `n_jobs` | <code>int &#124; None</code> | `None` |
| `pad` | <code>str</code> | `'auto'` |
| `method` | <code>Literal['fft', 'polyphase']</code> | `'fft'` |

### `ica`

Used by `ICA` and `run_ica`.

| Option | Type | General default |
| --- | --- | --- |
| `fit_only` | <code>bool</code> | `False` |
| `n_components` | <code>None &#124; int &#124; float</code> | `None` |
| `method` | <code>str</code> | `'picard'` |
| `random_state` | <code>None &#124; int</code> | `42` |
| `fit_params` | <code>dict &#124; None</code> | `None` |
| `ica_hp_freq` | <code>None &#124; float</code> | `1.0` |
| `ica_lp_freq` | <code>None &#124; float</code> | `None` |
| `resample_freq` | <code>int &#124; None</code> | `200` |
| `eog` | <code>bool</code> | `True` |
| `surrogate_eog_chs` | <code>None &#124; dict</code> | `None` |
| `eog_corr_thresh` | <code>float</code> | `0.5` |
| `ecg` | <code>bool</code> | `True` |
| `ecg_corr_thresh` | <code>float</code> | `0.5` |
| `emg` | <code>bool</code> | `False` |
| `emg_thresh` | <code>float</code> | `0.5` |
| `train` | <code>bool</code> | `False` |
| `train_freq` | <code>float</code> | `16.666` |
| `train_thresh` | <code>float</code> | `6.0` |
| `img_path` | <code>None &#124; str</code> | `None` |
| `fname` | <code>None &#124; str</code> | `None` |

### `events`

Used by `Events` and trigger selection in trial helpers.

| Option | Type | General default |
| --- | --- | --- |
| `stim_channel` | <code>str &#124; list[str] &#124; None</code> | `None` |
| `output` | <code>str</code> | `'onset'` |
| `consecutive` | <code>bool &#124; str</code> | `'increasing'` |
| `min_duration` | <code>float</code> | `0.0` |
| `shortest_event` | <code>int</code> | `2` |
| `mask` | <code>int &#124; None</code> | `None` |
| `uint_cast` | <code>bool</code> | `False` |
| `mask_type` | <code>str</code> | `'and'` |
| `initial_event` | <code>bool</code> | `False` |
| `verbose` | <code>bool &#124; str &#124; int &#124; None</code> | `None` |

### `physio`

Used by `PhysioCleaner` and `run_bio_preproc`.

| Option | Type | General default |
| --- | --- | --- |
| `ecg` | <code>None &#124; str &#124; list</code> | `None` |
| `resp` | <code>None &#124; str &#124; list</code> | `None` |
| `eog` | <code>None &#124; str &#124; list</code> | `None` |
| `emg` | <code>None &#124; str &#124; list</code> | `None` |

### `maxwell`

Used by `Maxwell`, `MultiBlockMaxwell`, and `run_maxwell`.

| Option | Type | General default |
| --- | --- | --- |
| `mw_coord_frame` | <code>str</code> | `'head'` |
| `mw_destination` | <code>Any</code> | `None` |
| `mw_calibration_file` | <code>str &#124; bool &#124; None</code> | `None` |
| `mw_cross_talk_file` | <code>str &#124; bool &#124; None</code> | `None` |
| `mw_st_duration` | <code>float &#124; None</code> | `None` |
| `mw_st_correlation` | <code>float</code> | `0.98` |

### `ransac`

Used by `EEGRANSAC`.

| Option | Type | General default |
| --- | --- | --- |
| `ransac_epoch_duration` | <code>int &#124; float</code> | `4` |
| `n_resample` | <code>int</code> | `50` |
| `min_channels` | <code>float</code> | `0.25` |
| `min_corr` | <code>float</code> | `0.75` |
| `unbroken_time` | <code>float</code> | `0.4` |
| `n_jobs` | <code>int</code> | `1` |
| `verbose` | <code>bool</code> | `False` |
| `random_state` | <code>int &#124; None</code> | `435656` |

### `rereference`

Used by `ReReference`.

| Option | Type | General default |
| --- | --- | --- |
| `ref_channels` | <code>str &#124; list[str] &#124; dict</code> | `'average'` |
| `projection` | <code>bool</code> | `False` |
| `ch_type` | <code>str &#124; list[str]</code> | `'auto'` |
| `forward` | <code>Any</code> | `None` |
| `joint` | <code>bool</code> | `False` |
| `verbose` | <code>bool &#124; str &#124; int &#124; None</code> | `False` |

### `epochs`

Used by `Epochs`.

| Option | Type | General default |
| --- | --- | --- |
| `tmin` | <code>float</code> | `-0.15` |
| `tmax` | <code>float</code> | `0.5` |
| `events` | <code>Any</code> | `None` |
| `event_id` | <code>None &#124; dict</code> | `None` |
| `baseline` | <code>None &#124; tuple</code> | `None` |
| `preload` | <code>bool</code> | `True` |
| `picks` | <code>Any</code> | `None` |
| `reject` | <code>dict &#124; None</code> | `None` |
| `flat` | <code>dict &#124; None</code> | `None` |
| `proj` | <code>bool &#124; str</code> | `True` |
| `reject_tmin` | <code>float &#124; None</code> | `None` |
| `reject_tmax` | <code>float &#124; None</code> | `None` |
| `detrend` | <code>int &#124; None</code> | `None` |
| `on_missing` | <code>str</code> | `'raise'` |
| `reject_by_annotation` | <code>bool</code> | `True` |
| `metadata` | <code>Any</code> | `None` |
| `event_repeated` | <code>str</code> | `'error'` |
| `verbose` | <code>bool &#124; str &#124; int &#124; None</code> | `None` |

### `forward_model`

Used by `ForwardModel` and optional arguments of `make_fwd`.

| Option | Type | General default |
| --- | --- | --- |
| `pick_dict` | <code>dict &#124; None</code> | `None` |
| `source` | <code>str</code> | `'surface'` |
| `redo_hdm` | <code>bool</code> | `True` |
| `spacing` | <code>str</code> | `'oct6'` |
| `source_ico` | <code>int</code> | `4` |
| `bem_conductivity` | <code>float &#124; Sequence[float]</code> | `(0.3,)` |
| `volume_pos` | <code>float</code> | `5.0` |
| `use_template_mri` | <code>bool</code> | `True` |
| `min_dist_src` | <code>float</code> | `5.0` |
| `meg` | <code>bool</code> | `True` |
| `eeg` | <code>bool</code> | `False` |
| `redo_bem` | <code>bool</code> | `False` |

### `spatial_filter`

Used by `SpatialFilter` and optional arguments of `comp_spatial_filters`.

| Option | Type | General default |
| --- | --- | --- |
| `fwd` | <code>Any</code> | `None` |
| `pick_dict` | <code>dict &#124; None</code> | `None` |
| `data_cov` | <code>Any</code> | `None` |
| `noise_cov` | <code>Any</code> | `None` |
| `empty_room` | <code>Any</code> | `None` |
| `nearest_empty_room` | <code>bool</code> | `False` |
| `chans2keep` | <code>list[str] &#124; None</code> | `None` |
| `lcmv_reg` | <code>float</code> | `0.05` |
| `lcmv_pick_ori` | <code>str &#124; None</code> | `'max-power'` |
| `lcmv_weight_norm` | <code>str &#124; None</code> | `'nai'` |
| `lcmv_reduce_rank` | <code>bool</code> | `False` |

### `source_reconstruction`

Used by `SourceReconstruction` and parcellation helpers.

| Option | Type | General default |
| --- | --- | --- |
| `filters` | <code>Any</code> | `None` |
| `return_parc` | <code>bool</code> | `False` |
| `label_mode` | <code>str</code> | `'pca_flip'` |
| `atlas` | <code>str</code> | `'glasser'` |
| `morph2fsaverage` | <code>bool</code> | `True` |

### `trf`

Used by `EpochTRF`, TRF epoch helpers, and fallback drift in trial helpers.

| Option | Type | General default |
| --- | --- | --- |
| `feature` | <code>str</code> | `'envelope'` |
| `audio_cutoff_hz` | <code>float</code> | `80.0` |
| `hw_delay_s` | <code>float</code> | `0.0` |
| `epoch_len_s` | <code>float</code> | `5.0` |
| `audio_channels` | <code>Sequence[str] &#124; None</code> | `None` |
| `alignment_kwargs` | <code>Mapping[str, Any] &#124; None</code> | `None` |
| `preserve_annotations` | <code>bool</code> | `True` |
| `on_alignment_error` | <code>Literal['raise', 'skip']</code> | `'raise'` |
| `verbose` | <code>bool</code> | `True` |
| `realign_without_audio` | <code>bool</code> | `False` |
| `fallback_drift_us_per_s` | <code>float</code> | `0.0` |

### `pipeline`

Used by `AlmKanal`.

| Option | Type | General default |
| --- | --- | --- |
| `pick_params` | <code>dict &#124; None</code> | `None` |

### `vpixx`

Used by `VPixxCleaner`, `VPixxConfig`, and eye loading/alignment helpers.

| Option | Type | General default |
| --- | --- | --- |
| `screen_resolution` | <code>tuple[int, int] &#124; None</code> | `None` |
| `screen_size` | <code>tuple[float, float] &#124; None</code> | `None` |
| `screen_distance` | <code>float &#124; None</code> | `None` |
| `calibration_model` | <code>str</code> | `'HV5'` |
| `calibration_eye` | <code>Literal['left', 'right']</code> | `'right'` |
| `blink_buffer_vpixx` | <code>tuple[float, float]</code> | `(0.05, 0.2)` |
| `digital_output_threshold` | <code>float</code> | `256.0` |
| `missing_value` | <code>float</code> | `9999.0` |
| `interpolate_blinks` | <code>bool</code> | `True` |
| `blink_buffer` | <code>tuple[float, float] &#124; None</code> | `None` |
| `convert_to_radians` | <code>bool</code> | `True` |
| `include_raw_vpixx_channels` | <code>bool</code> | `False` |
| `meg_stim_channel` | <code>str &#124; None</code> | `None` |
| `eye_stim_channel` | <code>str</code> | `'Digital Output'` |
| `meg_min_duration` | <code>float</code> | `0.0` |
| `meg_max_trigger` | <code>int &#124; None</code> | `None` |
| `concatenate_raws` | <code>bool</code> | `False` |
| `align` | <code>bool</code> | `True` |

### `ica_eog`

Used by `eog_ica_from_meg`.

| Option | Type | General default |
| --- | --- | --- |
| `left_eog_chs` | <code>list[str] &#124; None</code> | `None` |
| `right_eog_chs` | <code>list[str] &#124; None</code> | `None` |
| `threshold` | <code>float</code> | `0.5` |
| `tol` | <code>float</code> | `0.2` |

### `ica_train`

Used by `find_train_ica`.

| Option | Type | General default |
| --- | --- | --- |
| `duration` | <code>int</code> | `8` |
| `overlap` | <code>float</code> | `0.5` |
| `hmax` | <code>float</code> | `2.0` |

### `trials`

Used by `find_audio_trials` and `TRFSpanSpec.from_events`.

| Option | Type | General default |
| --- | --- | --- |
| `end_triggers` | <code>int &#124; Sequence[int] &#124; None</code> | `None` |
| `infer_missing_ends` | <code>bool</code> | `False` |
| `base_audio_path` | <code>Any</code> | `None` |

### `alignment`

Used by `estimate_raw_wav_alignment`.

| Option | Type | General default |
| --- | --- | --- |
| `sync_sfreq` | <code>float</code> | `500.0` |
| `audio_band` | <code>tuple[float, float]</code> | `(80.0, 400.0)` |
| `envelope_lowpass` | <code>float</code> | `30.0` |
| `window_s` | <code>float</code> | `10.0` |
| `step_s` | <code>float</code> | `5.0` |
| `max_lag_s` | <code>float</code> | `0.25` |
| `min_corr` | <code>float</code> | `0.3` |
| `min_anchors` | <code>int</code> | `5` |
| `reject_outliers` | <code>bool</code> | `True` |
| `warn_residual_rms_ms` | <code>float</code> | `10.0` |
| `verbose` | <code>bool</code> | `True` |

### `audio`

Used by `prepare_audio`.

| Option | Type | General default |
| --- | --- | --- |
| `feature` | <code>str</code> | `'envelope'` |
| `target_fs` | <code>float</code> | `100.0` |
| `cutoff_hz` | <code>float</code> | `80.0` |
| `n_mels` | <code>int</code> | `32` |
| `fmin` | <code>float</code> | `50.0` |
| `fmax` | <code>float</code> | `8000.0` |
