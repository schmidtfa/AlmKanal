"""TRF example: no recorded audio, with missing end triggers.

The same assumed rate serves two purposes:
1. make_spans estimates each missing end from WAV duration and drift.
2. EpochTRF resamples neural data using that drift. This is the only correction.

realign_without_audio=True belongs ONLY on EpochTRF. Pass no alignment_kwargs:
there is no recorded-audio correlation to configure. The WAV must begin at the
sound onset represented by the onset trigger; an initial offset cannot be
measured. Positive drift means t_raw = (1 + drift / 1e6) * t_wav, relative to
each trial onset. At +499 us/s, a 1000-s WAV spans 1000.499 s on the raw clock.

If end triggers exist, use end_triggers=99 (your actual code). Set
infer_missing_ends=False when all ends are present; the EpochTRF settings stay
the same. End triggers do not determine the assumed drift rate.
See trf_pipeline.py for the overview of all four examples.
"""

from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec


class TRFAssumedDriftPipe(Job):
    job_data_folder = 'data_meg'

    def run(
        self,
        subject_id: str,
        data_path: str,
        audio_path: str,
        hw_delay_s: float = 0.0165,
        epoch_len_s: float = 5.0,
        fallback_drift_us_per_s: float = 499.0,
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
                end_triggers=None,
                stim_channel='STI101',
                infer_missing_ends=True,
                base_audio_path=audio_path,
                # Estimate an endpoint; this does not resample any data.
                fallback_drift_us_per_s=fallback_drift_us_per_s,
            )

        ak = AlmKanal(
            pick_params=pick_dict,
            steps=[
                EpochTRF(
                    gen_span_spec=make_spans,
                    base_audio_path=audio_path,
                    audio_channels=None,
                    realign_without_audio=True,
                    # Use the SAME rate as in make_spans. Change the run()
                    # argument once to override both defaults, e.g. to 500.0.
                    fallback_drift_us_per_s=fallback_drift_us_per_s,
                    # Omit alignment_kwargs entirely in this mode.
                    feature=feature,  # Use 'envelope' or 'flux', not 'env_rms'.
                    # Advance neural data by the playback-to-ear delay AFTER
                    # drift correction; the WAV feature channels stay unchanged.
                    # Use 0 if the reference already represents ear arrival.
                    hw_delay_s=hw_delay_s,
                    epoch_len_s=epoch_len_s,
                    on_alignment_error='raise',
                ),
                # Add further filtering, downsampling, or source steps here.
            ],
        )
        epochs, report = ak.run(raw)

        # Reports label the rate as assumed; no offset or fit quality is measured.
        output_path = Path(self.full_output_path)
        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))
        joblib.dump((epochs, report), self.full_output_path)
