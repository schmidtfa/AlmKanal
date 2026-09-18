"""TRF example: recorded audio with onset and end triggers.

Choose the example that matches your recording:
- trf_pipeline.py: recorded audio and existing end triggers.
- trf_inferred_ends_pipeline.py: recorded audio, with missing end triggers.
- trf_assumed_drift_pipeline.py: no recorded audio; assume +499 us/s drift.
- trf_fixed_delay_pipeline.py: physical-delay correction without realignment.

Each file is a self-contained Job template. Adapt channel names, trigger codes,
WAV names, and input paths to your dataset. Span generation locates trials;
EpochTRF performs the clock correction and attaches WAV-derived features.
"""

from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec


class TRFPipe(Job):
    job_data_folder = 'data_meg'
    full_output_path: Path  # Supplied by plus_slurm when executing the job.

    def run(
        self,
        subject_id: str,
        data_path: str,
        audio_path: str,
        hw_delay_s: float = 0.0165,
        epoch_len_s: float = 5.0,
        audio_channels: tuple[str, ...] = ('AUDIO001',),
        feature: str = 'envelope',
    ) -> None:
        full_path = Path(data_path) / f'{subject_id}_raw.fif'
        raw = mne.io.read_raw(full_path, preload=True)

        # Preserve the stim and recorded audio channels for trial realignment.
        pick_dict = {
            'meg': True,
            'eog': True,
            'ecg': True,
            'eeg': False,
            'stim': True,
            'misc': True,
        }
        onset_trigger_to_wav = {
            11: 'story_a.wav',
            12: 'story_b.wav',
        }

        def make_spans(raw: mne.io.BaseRaw) -> TRFSpanSpec:
            # Repeated presentations receive distinct trial labels.
            return TRFSpanSpec.from_events(
                raw,
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
                    # Estimate onset offset AND drift from recorded audio.
                    audio_channels=audio_channels,
                    alignment_kwargs={
                        'window_s': 10.0,
                        'step_s': 5.0,
                        'min_corr': 0.3,
                        # Allow initial offset plus drift over the full WAV.
                        # At +499 us/s, 20 minutes adds about 0.6 s of lag.
                        'max_lag_s': 1.0,
                    },
                    # 'envelope' produces 'env_rms'; 'flux' produces 'flux'.
                    feature=feature,
                    # Realign first, then apply the physical delay. The default
                    # +0.0165 advances MEG to compensate the 16.5 ms air-tube
                    # delay; the subsequently added WAV feature is not shifted.
                    hw_delay_s=hw_delay_s,
                    epoch_len_s=epoch_len_s,
                    on_alignment_error='raise',
                ),
                # Keep sufficient audio bandwidth until EpochTRF has run.
                # Add filtering and downsampling afterward as needed.
            ],
        )
        epochs, report = ak.run(raw)

        output_path = Path(self.full_output_path)
        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))
        joblib.dump((epochs, report), self.full_output_path)

