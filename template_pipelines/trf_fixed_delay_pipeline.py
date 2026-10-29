"""TRF example: physical-delay correction with realignment disabled.

Existing onset/end triggers define the trials. EpochTRF attaches WAV-derived
features and applies hw_delay_s, without estimating offset or resampling for
clock drift. This is the default behavior when audio_channels=None and
realign_without_audio=False.

If ends are missing, make_spans can use infer_missing_ends=True,
end_triggers=None, base_audio_path=audio_path, and fallback_drift_us_per_s=499.0.
That ONLY estimates endpoints; it does not enable drift correction. To also
correct drift without audio, use trf_assumed_drift_pipeline.py.
See trf_pipeline.py for the overview of all four examples.
"""

from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec


class TRFFixedDelayPipe(Job):
    job_data_folder = 'data_meg'

    def run(
        self,
        subject_id: str,
        data_path: str,
        audio_path: str,
        hw_delay_s: float = 0.0165,
        epoch_len_s: float = 5.0,
        feature: str = 'envelope',
    ) -> None:
        full_path = Path(data_path) / f'{subject_id}_raw.fif'
        raw = mne.io.read_raw(full_path, preload=True)
        pick_dict = {'meg': True, 'eog': True, 'ecg': True, 'eeg': False, 'stim': True, 'misc': False}
        onset_trigger_to_wav = {11: 'story_a.wav', 12: 'story_b.wav'}

        def make_spans(current_raw: mne.io.BaseRaw) -> TRFSpanSpec:
            return TRFSpanSpec.from_events(
                current_raw,
                onset_trigger_to_wav=onset_trigger_to_wav,
                end_triggers=99,
                stim_channel='STI101',
            )

        ak = AlmKanal(
            pick_params=pick_dict,
            steps=[
                EpochTRF(
                    gen_span_spec=make_spans,
                    base_audio_path=audio_path,
                    # Explicitly show the defaults that disable realignment.
                    audio_channels=None,
                    realign_without_audio=False,
                    # Omit alignment_kwargs; no recorded audio is used.
                    feature=feature,  # Use 'envelope' or 'flux', not 'env_rms'.
                    # Positive values advance neural data relative to the WAV
                    # features to compensate the 16.5 ms playback-to-ear delay.
                    hw_delay_s=hw_delay_s,
                    epoch_len_s=epoch_len_s,
                ),
            ],
        )
        epochs, report = ak.run(raw)

        output_path = Path(self.full_output_path)
        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))
        joblib.dump((epochs, report), self.full_output_path)
