from pathlib import Path
from typing import Literal

import mne
from attrs import define, field

from almkanal.almkanal import AlmKanalStep
from almkanal.defaults import default_field, step_with_defaults
from almkanal.eye_utils.gaze_utils import VPixxConfig, align_eye_to_meg, load_eyetracking_data
from almkanal.info import AlmKanalInfo


@define
class VPixxCleaner(AlmKanalStep):
    """Load, clean, and align VPixx eye tracking with continuous M/EEG data.

    Omitted options use the active ``vpixx`` defaults at construction. The
    general profile requires screen geometry when converting gaze to radians;
    select the Salzburg profile or supply measured geometry for another setup.
    Screen dimensions and viewing distance are in metres. ``eye_path`` is
    required and must contain one path per recording when processing blocks.
    """

    # vpixx config
    eye_path: str | Path | list[str | Path]
    screen_resolution: tuple[int, int] | None = default_field('vpixx', 'screen_resolution')
    screen_size: tuple[float, float] | None = default_field('vpixx', 'screen_size')
    screen_distance: float | None = default_field('vpixx', 'screen_distance')
    calibration_model: str = default_field('vpixx', 'calibration_model')
    calibration_eye: Literal['left', 'right'] = default_field('vpixx', 'calibration_eye')
    blink_buffer_vpixx: tuple[float, float] = default_field('vpixx', 'blink_buffer_vpixx')
    digital_output_threshold: float = default_field('vpixx', 'digital_output_threshold')
    missing_value: float = default_field('vpixx', 'missing_value')
    # loading setup
    interpolate_blinks: bool = default_field('vpixx', 'interpolate_blinks')
    blink_buffer: tuple[float, float] | None = default_field('vpixx', 'blink_buffer')
    convert_to_radians: bool = default_field('vpixx', 'convert_to_radians')
    include_raw_vpixx_channels: bool = default_field('vpixx', 'include_raw_vpixx_channels')
    # align info
    meg_stim_channel: str | None = default_field('vpixx', 'meg_stim_channel')
    eye_stim_channel: str = default_field('vpixx', 'eye_stim_channel')
    meg_min_duration: float = default_field('vpixx', 'meg_min_duration')
    meg_max_trigger: int | None = default_field('vpixx', 'meg_max_trigger')
    concatenate_raws: bool = default_field('vpixx', 'concatenate_raws')
    must_be_before: tuple = ('MultiBlockMaxwell', 'Maxwell', 'Epochs', 'Events')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    @step_with_defaults
    def run(
        self,
        data: mne.io.BaseRaw | list[mne.io.BaseRaw],
        info: AlmKanalInfo,
    ) -> dict:
        config = VPixxConfig(
            self.screen_resolution,
            self.screen_size,
            self.screen_distance,
            self.calibration_model,
            self.calibration_eye,
            self.blink_buffer_vpixx,
            self.digital_output_threshold,
            self.missing_value,
        )
        single_raw = isinstance(data, mne.io.BaseRaw)

        if single_raw:
            raw_list: list[mne.io.BaseRaw] = [data]
        else:
            raw_list = list(data)

        eye_paths = [self.eye_path] if isinstance(self.eye_path, str | Path) else list(self.eye_path)

        if len(raw_list) != len(eye_paths):
            raise ValueError(
                f'Received {len(raw_list)} MEG recordings but '
                f'{len(eye_paths)} eye-tracking paths. '
                'Each MEG recording must have a corresponding eye-tracking file.'
            )
        raw_list_preproc = []
        for raw, eye_path in zip(raw_list, eye_paths):
            eye_data = load_eyetracking_data(
                eye_path=eye_path,
                config=config,
                interpolate_blinks=self.interpolate_blinks,
                blink_buffer=self.blink_buffer,
                convert_to_radians=self.convert_to_radians,
                include_raw_vpixx_channels=self.include_raw_vpixx_channels,
            )

            align_eye_to_meg(
                raw,
                eye_data,
                meg_stim_channel=self.meg_stim_channel,
                eye_stim_channel=self.eye_stim_channel,
                meg_min_duration=self.meg_min_duration,
                meg_max_trigger=self.meg_max_trigger,
            )

            raw.load_data()
            eye_data.load_data()

            if raw.info['sfreq'] != eye_data.info['sfreq']:
                raise ValueError(
                    'MEG and eye-tracking sampling frequencies differ after '
                    'alignment. Resample the eye-tracking data before adding it '
                    'to the MEG recording.'
                )

            if raw.n_times != eye_data.n_times:
                raise ValueError(
                    'MEG and eye-tracking recordings have different numbers of '
                    'samples. They cannot be combined with Raw.add_channels(). '
                    'Check temporal alignment and recording durations.'
                )

            raw.add_channels(
                [eye_data],
                force_update_info=True,
            )
            raw_list_preproc.append(raw)
        if single_raw:
            result_data = raw_list_preproc[0]
        elif not single_raw and self.concatenate_raws:
            result_data = mne.concatenate_raws(raw_list_preproc)
        elif not single_raw and not self.concatenate_raws:
            result_data = raw_list_preproc

        return {
            'data': result_data,
            'eye_info': {
                'screen_resolution': self.screen_resolution,
                'screen_size': self.screen_size,
                'screen_distance': self.screen_distance,
                'calibration_model': self.calibration_model,
                'calibration_eye': self.calibration_eye,
                'blink_buffer_vpixx': self.blink_buffer_vpixx,
                'digital_output_threshold': self.digital_output_threshold,
                'missing_value': self.missing_value,
                'interpolate_blinks': self.interpolate_blinks,
                'blink_buffer': self.blink_buffer,
                'convert_to_radians': self.convert_to_radians,
                'include_raw_vpixx_channels': self.include_raw_vpixx_channels,
                'meg_stim_channel': self.meg_stim_channel,
                'eye_stim_channel': self.eye_stim_channel,
                'meg_min_duration': self.meg_min_duration,
                'meg_max_trigger': self.meg_max_trigger,
                'concatenate_raws': self.concatenate_raws,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        pass
