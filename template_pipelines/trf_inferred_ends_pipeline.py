"""TRF example: recorded audio with missing end triggers.

make_spans uses WAV duration and an assumed drift to infer missing trial ends.
EpochTRF then MEASURES the onset offset and drift from recorded audio and
resamples the neural data. The endpoint prior does not fix the measured drift.

realign_without_audio belongs only to EpochTRF and is unnecessary here.
For a recording without audio channels, use trf_assumed_drift_pipeline.py.
See trf_pipeline.py for the overview of all four examples.
"""

from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec


class TRFInferredEndsPipe(Job):
    job_data_folder = 'data_meg'

    def run(
        self,
        subject_id: str,
        data_path: str,
        audio_path: str,
        hw_delay_s: float = 0.0165,
        epoch_len_s: float = 5.0,
        fallback_drift_us_per_s: float = 499.0,
        audio_channels: tuple[str, ...] = ('AUDIO001',),
        feature: str = 'envelope',
    ) -> None:
        full_path = Path(data_path) / f'{subject_id}_raw.fif'
        raw = mne.io.read_raw(full_path, preload=True)
        pick_dict = {'meg': True, 'eog': True, 'ecg': True, 'eeg': False, 'stim': True, 'misc': True}
        onset_trigger_to_wav = {11: 'story_a.wav', 12: 'story_b.wav'}

        def make_spans(current_raw: mne.io.BaseRaw) -> TRFSpanSpec:
            return TRFSpanSpec.from_events(
                current_raw,
                onset_trigger_to_wav=onset_trigger_to_wav,
                # If only SOME ends are missing, set this to the actual code
                # (e.g. 99). Existing end triggers will then take precedence.
                end_triggers=None,
                stim_channel='STI101',
                infer_missing_ends=True,
                base_audio_path=audio_path,
                fallback_drift_us_per_s=fallback_drift_us_per_s,
            )

        ak = AlmKanal(
            pick_params=pick_dict,
            steps=[
                EpochTRF(
                    gen_span_spec=make_spans,
                    base_audio_path=audio_path,
                    audio_channels=audio_channels,
                    # These settings control the recorded-audio correlation.
                    # End inference does not widen the lag search: include
                    # initial offset plus drift over the entire WAV here.
                    alignment_kwargs={
                        'window_s': 10.0,
                        'step_s': 5.0,
                        'min_corr': 0.3,
                        'max_lag_s': 1.0,
                    },
                    feature=feature,  # Use 'envelope' or 'flux', not 'env_rms'.
                    # Applied to neural data after measured clock correction.
                    hw_delay_s=hw_delay_s,
                    epoch_len_s=epoch_len_s,
                    on_alignment_error='raise',
                ),
                # Keep the recorded audio channels and their bandwidth until
                # EpochTRF has run. Filter and downsample afterward as needed.
            ],
        )
        epochs, report = ak.run(raw)

        output_path = Path(self.full_output_path)
        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))
        joblib.dump((epochs, report), self.full_output_path)
