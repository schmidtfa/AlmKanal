# Template pipeline for EOG preprocessing with AlmKanal
from pathlib import Path

import joblib
import mne
from cluster_jobs.meta_job import Job

from almkanal import AlmKanal
from almkanal.almkanal_steps.eog_utils import EOGCleaner


class EOGPipe(Job):
    job_data_folder = 'data_eog'

    def run(
        self,
        subject_id: str,
        eog_channels: tuple[str, str] = ('EOG001', 'EOG002'),
        find_blinks: bool = True,
        interpolate_blinks: bool = True,
        peak_threshold: float = 3.0,
        blink_window_ms: float = 300.0,
        check_and_correct_eog_polarity: bool = True,
        check_and_correct_eog_channel_assignment: bool = True,
        flat_threshold_v: float = 1e-8,
        max_amplitude_v: float | None = None,
        max_bad_fraction: float = 0.3,
        l_freq: float | None = 0.1,
        h_freq: float | None = 40.0,
    ) -> None:
        data_path = Path(self.full_input_path)
        files = sorted(Path(data_path).glob(f'{subject_id}*.fif'))

        if len(files) != 1:
            raise ValueError(
                f'Expected one recording for {subject_id}, '
                f'but found {len(files)}. '
                'Specify a folder containing only the intended recording.'
            )

        raw_eog = mne.io.read_raw_fif(files[0], preload=False)

        # Mark the supplied channels as EOG before channel selection.
        raw_eog.set_channel_types(
            {
                eog_channels[0]: 'eog',
                eog_channels[1]: 'eog',
            }
        )

        # Preserve stimulus and recorded audio channels.
        pick_dict = {
            'meg': False,
            'eog': True,
            'ecg': False,
            'eeg': False,
            'stim': True,
            'misc': True,
        }

        ak = AlmKanal(
            pick_params=pick_dict,
            steps=[
                EOGCleaner(
                    eog_channels=eog_channels,
                    find_blinks=find_blinks,
                    interpolate_blinks=interpolate_blinks,
                    peak_threshold=peak_threshold,
                    blink_window_ms=blink_window_ms,
                    check_and_correct_eog_polarity=check_and_correct_eog_polarity,
                    check_and_correct_eog_channel_assignment=check_and_correct_eog_channel_assignment,
                    flat_threshold_v=flat_threshold_v,
                    max_amplitude_v=max_amplitude_v,
                    max_bad_fraction=max_bad_fraction,
                    l_freq=l_freq,
                    h_freq=h_freq,
                ),
            ],
        )

        cleaned_raw_eog, report = ak.run(raw_eog)

        output_path = Path(self.full_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))

        joblib.dump((cleaned_raw_eog, report), output_path)
