## Prepare your inputs

`VPixxCleaner` loads eye-tracking data, applies supported cleaning options, aligns it to a continuous recording using the configured trigger channels, and adds eye-tracking channels to the recording.

You need a supported eye-tracking file for each raw recording, identifiable synchronization events, and the correct screen and calibration settings. The guide below is an API example, not a validated recipe for every VPixx export format or acquisition setup.

## Configure the step

`VPixxCleaner` uses the active `vpixx` [defaults profile](defaults.md). The generic
profile leaves screen geometry unset; supply it as below, or call
`set_defaults("salzburg")` before creating the step to use the Salzburg setup.
Screen dimensions and viewing distance are in metres. Pixel-only processing
with `convert_to_radians=False` does not require screen geometry.

```python
import mne
from almkanal import AlmKanal, VPixxCleaner

raw = mne.io.read_raw_fif("your_recording_raw.fif", preload=True)

pipeline = AlmKanal([
    VPixxCleaner(
        eye_path="path/to/your_supported_vpixx_file",
        meg_stim_channel="STI101",       # Use your actual trigger channel.
        eye_stim_channel="Digital Output",
        screen_resolution=(1920, 1080), # Example setup: replace as needed.
        screen_size=(0.61, 0.34),       # Screen dimensions in metres.
        screen_distance=0.82,          # Viewing distance in metres.
        calibration_eye="right",
        interpolate_blinks=True,
        convert_to_radians=True,
    ),
])

aligned_raw, report = pipeline.run(raw.copy())
pipeline.generate_json("eye_preprocessing.json")
```

Keep the synchronization channels available until alignment has completed. The example's channel names and dimensions must be replaced when your setup differs.

To reuse settings across scripts and standalone helpers:

```python
from almkanal import set_defaults, VPixxCleaner
from almkanal.eye_utils.gaze_utils import load_eyetracking_data

set_defaults("salzburg", vpixx={"blink_buffer_vpixx": (0.05, 0.25)})
step = VPixxCleaner("eye.mat")
eye = load_eyetracking_data("eye.mat")
```

The step captures settings at construction; the standalone loader reads the
active profile at call time. Explicit arguments override profile values. The
loader obtains its sampling rate from the MAT-file timestamps.

## Check the result

Check event correspondences and timing across the recording, channel names and units, blink handling, and retained samples. The implementation checks equal sampling rates and sample counts before adding eye channels. Matching dimensions alone are not evidence of correct synchronization.

## Work with recording blocks

For several raw blocks, supply an equally long list of eye-tracking paths. Set `concatenate_raws=True` only when concatenation is wanted; otherwise the step returns a list and subsequent steps must support it. This is important when you want to do `MultiBlockMaxwell` afterwards. If you are not using MEG data you probably want to `concatenate_raws=True` to avoid errors with subsequent steps. Currently only `VPIXXCleaner` and `MultiBlockMaxwell` accept lists of `Raw` or `Epoch` objects.

## Respect ordering requirements

When present together, `VPixxCleaner` is declared to run before `MultiBlockMaxwell`, `Maxwell`, `Epochs`, and `Events`. This step requires continuous data; it is not a way to align eye-tracking files to an arbitrary pre-existing `Epochs` object.

Review the loader and alignment implementation for file-format details and acquisition assumptions before using it on a new recording system.
