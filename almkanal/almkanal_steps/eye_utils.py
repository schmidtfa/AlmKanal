from pathlib import Path
from typing import Literal

import mne
from attrs import define, field

from almkanal import AlmKanalStep
from almkanal.eye_utils.gaze_utils import VPixxConfig, align_eye_to_meg, load_eyetracking_data
from almkanal.info import AlmKanalInfo


@define
class VPixxCleaner(AlmKanalStep):
    """This class implements the eye cleaning tools by the ocular tracking gang in AlmKanal"""

    # vpixx config
    eye_path: str | Path
    screen_resolution: tuple[int, int] = (1920, 1080)
    screen_size: tuple[float, float] = (0.61, 0.34)
    screen_distance: float = 0.82
    calibration_model: str = 'HV5'
    calibration_eye: Literal['left', 'right'] = 'right'
    blink_buffer_vpixx: tuple[float, float] = (0.05, 0.2)
    digital_output_threshold: float = 256.0
    missing_value: float = 9999.0
    # loading setup
    interpolate_blinks: bool = True
    blink_buffer: tuple[float, float] | None = None
    convert_to_radians: bool = True
    include_raw_vpixx_channels: bool = False
    # align info
    meg_stim_channel: str = 'STI101'
    eye_stim_channel: str = 'Digital Output'
    meg_min_duration: float = 0.002
    meg_max_trigger: int = 4096

    must_be_before: tuple = ('Epochs', 'Events')
    must_be_after: tuple = ()
    allow_repeated: bool = field(default=False, init=False)

    def run(
        self,
        data: mne.io.BaseRaw,
        info: AlmKanalInfo,
    ) -> dict:
        eye_data = load_eyetracking_data(
            eye_path=self.eye_path,
            config=VPixxConfig(
                self.screen_resolution,
                self.screen_size,
                self.screen_distance,
                self.calibration_model,
                self.calibration_eye,
                self.blink_buffer_vpixx,  # why are there two blink buffers?
                self.digital_output_threshold,
                self.missing_value,
            ),
            interpolate_blinks=self.interpolate_blinks,
            blink_buffer=self.blink_buffer,
            convert_to_radians=self.convert_to_radians,
            include_raw_vpixx_channels=self.include_raw_vpixx_channels,
        )

        align_eye_to_meg(
            data,
            eye_data,
            meg_stim_channel=self.meg_stim_channel,
            eye_stim_channel=self.eye_stim_channel,
            meg_min_duration=self.meg_min_duration,
            meg_max_trigger=self.meg_max_trigger,
        )

        data.load_data()
        eye_data.load_data()

        if data.info['sfreq'] != eye_data.info['sfreq']:
            raise ValueError(
                'MEG and eye-tracking sampling frequencies differ after '
                'alignment. Resample the eye-tracking data before adding it '
                'to the MEG recording.'
            )

        if data.n_times != eye_data.n_times:
            raise ValueError(
                'MEG and eye-tracking recordings have different numbers of '
                'samples. They cannot be combined with Raw.add_channels(). '
                'Check temporal alignment and recording durations.'
            )

        data.add_channels(
            [eye_data],
            force_update_info=True,
        )

        return {
            'data': data,
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
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: AlmKanalInfo) -> None:
        pass
