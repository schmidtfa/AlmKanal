from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Literal

import attrs
import mne
import numpy as np
import pandas as pd
from attrs import define, field

from almkanal.almkanal import AlmKanalStep
from almkanal.stim_utils.alignment_utils import (
    DEFAULT_FALLBACK_DRIFT_US_PER_S,
    apply_raw_wav_alignment,
    assume_raw_wav_alignment,
    estimate_raw_wav_alignment,
    find_audio_trials,
    summarize_alignments,
)
from almkanal.stim_utils.audio_utils import prepare_audio

Spans = dict[str, tuple[int, int | None]]
MetaMap = dict[str, Mapping[str, Any]]


def _validate_spans(_inst: Any, _attr: attrs.Attribute[Spans], spans: Spans) -> None:
    on_off_dim = 2

    if not isinstance(spans, dict):
        raise TypeError('spans_by_label must be a dict')
    for k, v in spans.items():
        if not isinstance(k, str):
            raise TypeError(f'label must be str, got {type(k)}')
        if (not isinstance(v, tuple)) or len(v) != on_off_dim:
            raise TypeError(f'span for {k} must be a 2-tuple (on, off)')
        on, off = v
        if not isinstance(on, int | np.integer):
            raise TypeError(f'onset for {k} must be int (raw samples)')
        if off is not None and not isinstance(off, int | np.integer):
            raise TypeError(f'offset for {k} must be int or None')
        if off is not None and int(off) <= int(on):
            # allow equality only if you intentionally want zero-length (we usually don't)
            raise ValueError(f'offset must be > onset for {k} (got {on}, {off})')


def _validate_meta(_inst: Any, _attr: attrs.Attribute[MetaMap], meta: MetaMap) -> None:
    if not isinstance(meta, dict):
        raise TypeError('metadata_by_label must be a dict')
    for k, v in meta.items():
        if not isinstance(k, str):
            raise TypeError('metadata keys must be labels (str)')
        if not hasattr(v, 'items'):
            raise TypeError(f'metadata for {k} must be a mapping (key -> value)')


@define
class TRFSpanSpec:
    """
    Minimal spec for TRF epoch building.
    - spans_by_label: REQUIRED raw-sample spans
    - metadata_by_label: OPTIONAL per-label metadata to merge into epoch metadata
                         values may be scalars (broadcast to all epochs of that label)
                         or sequences of length n_epochs for that label
    - wav_by_label: OPTIONAL WAV paths, relative to base_audio_path or absolute.
                   Distinct trial labels may reference the same WAV.
    """

    spans_by_label: Spans = field(validator=_validate_spans)
    metadata_by_label: MetaMap = field(factory=dict, validator=_validate_meta)
    wav_by_label: Mapping[str, str | Path] = field(factory=dict)

    def labels(self) -> list:
        # preserve insertion order from user’s dict (Python 3.7+)
        return list(self.spans_by_label.keys())

    @classmethod
    def from_events(
        cls,
        raw: mne.io.BaseRaw,
        onset_trigger_to_wav: Mapping[int, str | Path],
        end_triggers: int | Sequence[int] | None = None,
        *,
        stim_channel: str | None = None,
        infer_missing_ends: bool = False,
        base_audio_path: str | Path | None = None,
        fallback_drift_us_per_s: float = DEFAULT_FALLBACK_DRIFT_US_PER_S,
    ) -> TRFSpanSpec:
        """Build ordered, uniquely labelled trials, including repeated WAVs.

        Set infer_missing_ends=True to replace missing end triggers with WAV
        duration * (1 + fallback_drift_us_per_s / 1e6), on the raw clock.
        The default prior is +499 us/s; this only defines the span. EpochTRF
        estimates actual offset and drift when audio_channels are supplied.
        Without recorded audio, enable realign_without_audio on EpochTRF to
        also resample the neural data, using the same fallback_drift_us_per_s.
        Pass base_audio_path for relative WAV paths, and end_triggers=None
        when no end-trigger codes exist. Actual end triggers take precedence.
        End-inference provenance is retained in epoch metadata and reports;
        the Raw trigger channel is not modified.
        """
        trials = find_audio_trials(
            raw,
            onset_trigger_to_wav,
            end_triggers,
            stim_channel=stim_channel,
            infer_missing_ends=infer_missing_ends,
            base_audio_path=base_audio_path,
            fallback_drift_us_per_s=fallback_drift_us_per_s,
        )
        spans: Spans = {}
        metadata: MetaMap = {}
        wavs: dict[str, str | Path] = {}
        for index, trial in enumerate(trials):
            label = f'trial_{index:03d}'
            spans[label] = (trial['onset_sample'], trial['end_sample'])
            wavs[label] = trial['wav_file']
            metadata[label] = {
                'stimulus': Path(trial['wav_file']).stem,
                'onset_code': trial['onset_code'],
                'end_code': trial['end_code'],
            }
            if infer_missing_ends:
                metadata[label] = {
                    **metadata[label],
                    'end_inferred': trial.get('end_inferred', False),
                    'end_inference_drift_us_per_s': trial.get('end_inference_drift_us_per_s'),
                    'end_inference_wav_duration_s': trial.get('end_inference_wav_duration_s'),
                }
        return cls(spans, metadata, wavs)


def _aligned_segment(
    raw: mne.io.BaseRaw,
    on_samp: int,
    off_samp: int | None,
    wav: Path,
    delay_samples: int,
    audio_channels: Sequence[str] | None,
    fallback_drift_us_per_s: float,
    alignment_kwargs: Mapping[str, Any],
    preserve_annotations: bool,
    verbose: bool,
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    sfreq = float(raw.info['sfreq'])
    if audio_channels is None:
        alignment = assume_raw_wav_alignment(wav, sfreq, fallback_drift_us_per_s=fallback_drift_us_per_s)
    else:
        trial = raw.copy().crop(
            tmin=(on_samp - raw.first_samp) / sfreq,
            tmax=None if off_samp is None else (off_samp - raw.first_samp) / sfreq,
        )
        alignment = estimate_raw_wav_alignment(
            trial,
            wav,
            audio_channels=audio_channels,
            verbose=verbose,
            **alignment_kwargs,
        )
        alignment = {**alignment, 'alignment_method': 'audio'}
    before = max(0, -delay_samples)
    after = max(0, delay_samples)
    aligned = apply_raw_wav_alignment(
        raw,
        on_samp,
        alignment,
        padding_s=(before / sfreq, after / sfreq),
        preserve_annotations=preserve_annotations,
        verbose=verbose,
    )
    # Apply the physical delay only AFTER correcting the recording clock.
    # output_neural(t) = aligned_neural(t + hw_delay_s): selecting later
    # samples advances neural events. The default +0.0165 compensates the
    # playback-to-ear delay in the air tubes. WAV features are attached afterward.
    start = before + delay_samples
    n_samples = int(round(alignment['wav_duration_s'] * sfreq))
    aligned.crop(tmin=start / sfreq, tmax=(start + n_samples - 1) / sfreq)
    return aligned, alignment


def _build_trf_epochs(  # noqa: C901, PLR0915, PLR0912
    raw: mne.io.BaseRaw,
    spec: TRFSpanSpec,
    base_audio_path: str | Path,
    *,
    feature: str = 'envelope',
    audio_cutoff_hz: float = 80.0,
    hw_delay_s: float = 0.0165,
    epoch_len_s: float = 5.0,
    wav_ext: str = '.wav',
    audio_channels: Sequence[str] | None = None,
    realign_without_audio: bool = False,
    fallback_drift_us_per_s: float = DEFAULT_FALLBACK_DRIFT_US_PER_S,
    alignment_kwargs: Mapping[str, Any] | None = None,
    preserve_annotations: bool = True,
    on_alignment_error: Literal['raise', 'skip'] = 'raise',
    verbose: bool = True,
) -> tuple[mne.Epochs, dict[str, Any]]:
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError('EpochTRF requires continuous mne.io.BaseRaw input.')
    if not np.isfinite(hw_delay_s) or not np.isfinite(epoch_len_s) or epoch_len_s <= 0:
        raise ValueError('hw_delay_s must be finite and epoch_len_s must be finite and positive.')
    if audio_channels is not None and (isinstance(audio_channels, str) or not audio_channels):
        raise ValueError('audio_channels must contain at least one recorded audio channel, or be None.')
    if alignment_kwargs and audio_channels is None:
        raise ValueError('alignment_kwargs requires audio_channels to enable realignment.')
    if realign_without_audio and (not np.isfinite(fallback_drift_us_per_s) or 1.0 + fallback_drift_us_per_s / 1e6 <= 0):
        raise ValueError('fallback_drift_us_per_s must be finite and yield a positive clock slope (> -1000000).')
    if on_alignment_error not in {'raise', 'skip'}:
        raise ValueError("on_alignment_error must be either 'raise' or 'skip'.")
    if hw_delay_s < 0:
        warnings.warn(
            'Negative hw_delay_s delays MEG/EEG relative to the unchanged WAV feature channels. '
            'This adds to a playback-to-ear delay instead of compensating it. The default '
            '+0.0165 s advances neural events to compensate the 16.5 ms air-tube delay. '
            'Check the sign of your physical-delay correction.',
            UserWarning,
            stacklevel=3,
        )
    sfreq = float(raw.info['sfreq'])
    first = int(raw.first_samp)
    delay_samples = int(round(hw_delay_s * sfreq))
    applied_delay_s = delay_samples / sfreq
    base_audio_path = Path(base_audio_path)
    epochs_list: list[mne.Epochs] = []
    label_to_code: dict[str, int] = {}
    trial_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    end_inference_drift_counts: dict[str, int] = {}

    for label in spec.labels():
        on_samp, off_samp = spec.spans_by_label[label]
        trial_metadata = spec.metadata_by_label.get(label, {})
        end_inferred = bool(trial_metadata.get('end_inferred', False))
        if end_inferred:
            rate = str(float(trial_metadata['end_inference_drift_us_per_s']))
            end_inference_drift_counts[rate] = end_inference_drift_counts.get(rate, 0) + 1
        wav = Path(spec.wav_by_label.get(label, Path(label).with_suffix(wav_ext)))
        if not wav.is_absolute():
            wav = base_audio_path / wav
        record = {
            'label': label,
            'wav_file': str(wav),
            'original_onset_sample': int(on_samp),
            'original_end_sample': None if end_inferred or off_samp is None else int(off_samp),
            'inferred_end_sample': int(off_samp) if end_inferred and off_samp is not None else None,
            'end_inferred': end_inferred,
            'end_inference_drift_us_per_s': trial_metadata.get('end_inference_drift_us_per_s'),
            'end_inference_wav_duration_s': trial_metadata.get('end_inference_wav_duration_s'),
        }
        alignment = None
        if audio_channels is not None or realign_without_audio:
            try:
                seg, alignment = _aligned_segment(
                    raw,
                    int(on_samp),
                    off_samp,
                    wav,
                    delay_samples,
                    audio_channels,
                    fallback_drift_us_per_s,
                    dict(alignment_kwargs or {}),
                    preserve_annotations,
                    verbose,
                )
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({**record, 'error': str(error)})
                if on_alignment_error == 'raise':
                    raise RuntimeError(f'Audio alignment failed for {label} (WAV={wav}).') from error
                warnings.warn(f'Skipping audio trial {label}: {error}', stacklevel=2)
                continue
            record.update(alignment)
            record['n_epochs'] = 0
            trial_records.append(record)

        audio_data, _t, audio_names, fs_aud = prepare_audio(
            audio_path=str(wav),
            feature=feature,
            target_fs=sfreq,
            cutoff_hz=audio_cutoff_hz,
        )
        if alignment is not None:
            desired_n = min(audio_data.shape[1], seg.n_times)
            slope = alignment['clock_slope']
            t_seg_on = (int(on_samp) - first) / sfreq + alignment['offset_s'] + slope * applied_delay_s
        else:
            if int(on_samp) < first:
                continue
            n_audio = audio_data.shape[1]
            if off_samp is not None:
                n_audio = min(n_audio, int(off_samp) - int(on_samp))
            start = int(on_samp) - first + delay_samples
            # If the physical correction extends beyond the recording, retain
            # only the matching audio/recording overlap.
            audio_start = max(0, -start)
            desired_n = min(n_audio - audio_start, raw.n_times - max(0, start))
            if desired_n <= 0:
                continue
            start = max(0, start)
            audio_data = audio_data[:, audio_start : audio_start + desired_n]
            seg = raw.copy().crop(tmin=start / sfreq, tmax=(start + desired_n - 1) / sfreq)
            if not preserve_annotations:
                seg.set_annotations(mne.Annotations([], [], []))
            t_seg_on = start / sfreq
            slope = 1.0

        n_epoch_samples = int(round(epoch_len_s * sfreq))
        if n_epoch_samples < 1:
            raise ValueError('epoch_len_s must span at least one sample.')
        if desired_n < n_epoch_samples:
            continue
        audio_data = audio_data[:, :desired_n]
        seg.crop(tmin=0, tmax=(desired_n - 1) / sfreq).load_data()
        info_aud = mne.create_info(audio_names, float(fs_aud), ['misc'] * len(audio_names))
        seg.add_channels([mne.io.RawArray(audio_data, info_aud, verbose=verbose)], force_update_info=True)

        label_code = len(label_to_code) + 1
        ep = mne.make_fixed_length_epochs(
            seg,
            duration=epoch_len_s,
            overlap=0,
            preload=True,
            id=label_code,
            reject_by_annotation=True,
            verbose=verbose,
        )
        if not len(ep):
            continue
        label_to_code[label] = label_code
        # Derive times from the retained events; annotation rejection may leave
        # gaps, so arange(len(ep)) would mislabel later epochs.
        local_on_s = (ep.events[:, 0] - seg.first_samp) / sfreq
        meta = {
            'label': [label] * len(ep),
            'segment_on_s': [t_seg_on] * len(ep),
            'segment_off_s': [t_seg_on + slope * (desired_n - 1) / sfreq] * len(ep),
            't_on': t_seg_on + slope * local_on_s,
            't_off': t_seg_on + slope * np.minimum(local_on_s + epoch_len_s, (desired_n - 1) / sfreq),
            'epoch_index_in_segment': ep.selection,
        }
        if alignment is not None:
            record['n_epochs'] = len(ep)
            meta.update(
                {
                    'wav_t_on': local_on_s,
                    'wav_t_off': local_on_s + epoch_len_s,
                    'alignment_offset_s': [alignment['offset_s']] * len(ep),
                    'alignment_method': [alignment['alignment_method']] * len(ep),
                    'clock_slope': [slope] * len(ep),
                    'drift_us_per_s': [alignment['drift_us_per_s']] * len(ep),
                    'hw_delay_s': [hw_delay_s] * len(ep),
                    'applied_hw_delay_s': [applied_delay_s] * len(ep),
                }
            )
        for key, value in spec.metadata_by_label.get(label, {}).items():
            if hasattr(value, '__len__') and not isinstance(value, str | bytes) and len(value) == len(ep):
                meta[key] = list(value)
            else:
                meta[key] = [value] * len(ep)
        ep.event_id = {label: label_code}
        ep.metadata = pd.DataFrame(meta)
        epochs_list.append(ep)

    if not epochs_list:
        raise RuntimeError('No epochs produced. Check spans/audio paths/durations and alignment failures.')
    epochs_all = mne.concatenate_epochs(epochs_list, on_mismatch='warn', verbose=verbose)
    epochs_all.event_id = label_to_code
    alignment_info = {
        'trials': trial_records,
        'failures': failures,
        'n_trials_found': len(spec.spans_by_label),
        'n_trials_aligned': len(trial_records),
        'n_trials_failed': len(failures),
        'n_trials_epoched': len(epochs_list),
        'n_trials_end_inferred': sum(end_inference_drift_counts.values()),
        'end_inference_drift_counts': end_inference_drift_counts,
        'summary': summarize_alignments(trial_records),
    }
    return epochs_all, alignment_info


def build_trf_epochs(
    raw: mne.io.BaseRaw,
    spec: TRFSpanSpec,
    base_audio_path: str | Path,
    *,
    feature: str = 'envelope',
    audio_cutoff_hz: float = 80.0,
    hw_delay_s: float = 0.0165,
    epoch_len_s: float = 5.0,
    wav_ext: str = '.wav',
    audio_channels: Sequence[str] | None = None,
    realign_without_audio: bool = False,
    fallback_drift_us_per_s: float = DEFAULT_FALLBACK_DRIFT_US_PER_S,
    alignment_kwargs: Mapping[str, Any] | None = None,
    preserve_annotations: bool = True,
    on_alignment_error: Literal['raise', 'skip'] = 'raise',
    verbose: bool = True,
) -> mne.Epochs:
    """Build TRF epochs, optionally realigning before physical-delay correction.

    Supplying audio_channels enables per-trial offset and drift estimation.
    With audio_channels=None, realign_without_audio=True instead resamples using
    WAV duration and fallback_drift_us_per_s (default +499 us/s). This assumes
    the trial onset is WAV time zero; no onset offset is measured. Positive
    drift means t_raw = (1 + drift / 1e6) * t_wav relative to the trial onset.
    Recorded audio takes precedence when supplied, and failed audio fits follow
    on_alignment_error without falling back to an assumed drift.
    hw_delay_s always shifts neural data relative to WAV features, after any
    realignment, rounded to the nearest sample on the corrected clock.
    Positive values advance neural events; the default +0.0165 s compensates
    the 16.5 ms playback-to-ear delay when the audio reference precedes the
    air tubes. Negative values add lag and emit a warning. The added WAV
    feature channels are never delay-shifted.

    For long stimuli, increase alignment_kwargs['max_lag_s'] to accommodate
    initial offset plus drift accumulated over the full WAV duration. See
    estimate_raw_wav_alignment for search-range guidance and fit warnings.
    """
    epochs, _ = _build_trf_epochs(
        raw,
        spec,
        base_audio_path,
        feature=feature,
        audio_cutoff_hz=audio_cutoff_hz,
        hw_delay_s=hw_delay_s,
        epoch_len_s=epoch_len_s,
        wav_ext=wav_ext,
        audio_channels=audio_channels,
        realign_without_audio=realign_without_audio,
        fallback_drift_us_per_s=fallback_drift_us_per_s,
        alignment_kwargs=alignment_kwargs,
        preserve_annotations=preserve_annotations,
        on_alignment_error=on_alignment_error,
        verbose=verbose,
    )
    return epochs


@define
class EpochTRF(AlmKanalStep):
    """Attach WAV features and epoch trials, with optional audio realignment.

    audio_channels=None retains fixed-delay-only processing unless
    realign_without_audio=True. That opt-in mode uses the original WAV duration
    and fallback_drift_us_per_s (default +499 us/s) to resample neural data,
    assuming WAV time zero at each trial onset: t_raw = (1 + drift / 1e6) * t_wav.
    It cannot measure an onset offset or confirm the assumed drift. The original
    end sample does not determine the drift; the WAV defines output duration.
    Recorded audio takes precedence when audio_channels are supplied and must
    still be present with sufficient bandwidth for alignment. Failed audio fits
    follow on_alignment_error without falling back to an assumed drift.
    Each trial is first corrected to WAV time, then hw_delay_s is applied to
    neural data before attaching the unchanged WAV feature channels. Positive
    values advance neural events relative to those features; the default
    +0.0165 s compensates the 16.5 ms playback-to-ear delay when the audio
    reference precedes the air tubes, preserving the brain's response latency.
    Use zero if the reference already captures sound arrival at the ears.
    Negative values add lag and emit a warning for this sign convention.
    Physical delays are rounded to the nearest sample on the corrected clock.
    Realignment retains recording margins to preserve the full WAV duration;
    insufficient recording coverage is an alignment error.

    The alignment search uses a fixed +/-max_lag_s range around zero for each
    window. Set alignment_kwargs['max_lag_s'] large enough for the initial
    offset plus accumulated drift over the FULL stimulus: 1260 s at +500 us/s
    adds 0.63 s, so a 1.0 s search is more appropriate than the 0.25 s default
    for a small initial offset. Missing-end inference does not adapt this range.
    Potentially unreliable fits emit RuntimeWarning, including with
    verbose=False, but are still applied; on_alignment_error only handles errors.
    alignment_kwargs['warn_residual_rms_ms'] sets the residual RMS warning
    threshold (default 10 ms). See estimate_raw_wav_alignment for all checks.
    """

    gen_span_spec: Callable
    base_audio_path: str | Path
    feature: str = 'envelope'
    audio_cutoff_hz: float = 80.0
    hw_delay_s: float = 0.0165
    epoch_len_s: float = 5.0
    audio_channels: Sequence[str] | None = None
    alignment_kwargs: Mapping[str, Any] | None = None
    preserve_annotations: bool = True
    on_alignment_error: Literal['raise', 'skip'] = 'raise'
    verbose: bool = True

    realign_without_audio: bool = field(default=False, kw_only=True)
    fallback_drift_us_per_s: float = field(default=DEFAULT_FALLBACK_DRIFT_US_PER_S, kw_only=True)

    must_be_before: tuple = ()
    must_be_after: tuple = ()

    def run(self, data: mne.io.BaseRaw, info: dict) -> dict:
        spec: TRFSpanSpec = self.gen_span_spec(data)
        sfreq = float(data.info['sfreq'])
        epochs, alignment_info = _build_trf_epochs(
            raw=data,
            spec=spec,
            base_audio_path=self.base_audio_path,
            feature=self.feature,
            audio_cutoff_hz=self.audio_cutoff_hz,
            hw_delay_s=self.hw_delay_s,
            epoch_len_s=self.epoch_len_s,
            audio_channels=self.audio_channels,
            realign_without_audio=self.realign_without_audio,
            fallback_drift_us_per_s=self.fallback_drift_us_per_s,
            alignment_kwargs=self.alignment_kwargs,
            preserve_annotations=self.preserve_annotations,
            on_alignment_error=self.on_alignment_error,
            verbose=self.verbose,
        )
        alignment_settings = {}
        if self.audio_channels is not None:
            alignment_settings = {
                name: parameter.default
                for name, parameter in signature(estimate_raw_wav_alignment).parameters.items()
                if parameter.default is not Parameter.empty and name != 'verbose'
            }
            alignment_settings.update(self.alignment_kwargs or {})
        return {
            'data': epochs,
            'TRF_info': {
                'spans_by_label': spec.spans_by_label,
                'wav_by_label': {label: str(path) for label, path in spec.wav_by_label.items()},
                'feature': self.feature,
                'audio_cutoff_hz': self.audio_cutoff_hz,
                'hw_delay_s': self.hw_delay_s,
                'applied_hw_delay_s': round(self.hw_delay_s * sfreq) / sfreq,
                'epoch_len_s': self.epoch_len_s,
                'realign_audio': self.audio_channels is not None or self.realign_without_audio,
                'alignment_method': (
                    'audio'
                    if self.audio_channels is not None
                    else 'assumed_drift'
                    if self.realign_without_audio
                    else None
                ),
                'realign_without_audio': self.realign_without_audio,
                'fallback_drift_us_per_s': self.fallback_drift_us_per_s,
                'audio_channels': None if self.audio_channels is None else list(self.audio_channels),
                'alignment_kwargs': alignment_settings,
                'preserve_annotations': self.preserve_annotations,
                'on_alignment_error': self.on_alignment_error,
                'alignment_info': alignment_info,
            },
        }

    def reports(self, data: mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        report.add_epochs(data, title='Epoched (TRF)')
        trf_info = info['EpochTRF']['TRF_info']
        if trf_info['realign_audio']:
            alignment = trf_info['alignment_info']
            columns = {
                'label': 'Trial',
                'alignment_method': 'Alignment method',
                'end_inferred': 'End inferred',
                'end_inference_drift_us_per_s': 'End prior (µs/s)',
                'offset_s': 'Offset (s)',
                'drift_us_per_s': 'Drift (µs/s)',
                'total_drift_ms': 'Total drift (ms)',
                'residual_rms_ms': 'Residual RMS (ms)',
                'median_correlation': 'Median r',
                'n_anchor_inliers': 'Inlier anchors',
                'n_epochs': 'Epochs',
            }
            table = pd.DataFrame(alignment['trials']).reindex(columns=list(columns)).rename(columns=columns)
            if trf_info.get('alignment_method') == 'assumed_drift':
                method = (
                    f'<p>Assumed drift correction: {trf_info["fallback_drift_us_per_s"]:g} µs/s. '
                    'Neural data were resampled using the original WAV duration, assuming WAV time zero '
                    'at the trial onset (zero offset). No recorded audio was used; drift and offset '
                    'were not measured, and residual/correlation diagnostics are unavailable.</p>'
                )
            else:
                method = '<p>Alignment offset and drift were estimated from recorded audio.</p>'
            html = (
                f'<p>Aligned {alignment["n_trials_aligned"]} of {alignment["n_trials_found"]} trials; '
                f'{alignment["n_trials_failed"]} failed. Physical-delay correction after realignment: '
                f'{trf_info["applied_hw_delay_s"] * 1000:g} ms '
                f'(requested {self.hw_delay_s * 1000:g} ms). '
                'WAV feature channels were left unchanged; positive values advance neural events '
                'to compensate playback-to-ear delay, and negative values add lag.</p>'
                f'<p>{alignment["n_trials_end_inferred"]} trial ends were inferred from WAV duration '
                'and an assumed drift rate.</p>'
                + method
                + table.to_html(index=False, escape=True, float_format=lambda value: f'{value:.6g}')
            )
            if alignment['failures']:
                html += pd.DataFrame(alignment['failures']).to_html(index=False, escape=True)
            report.add_html(html, title='TRF audio realignment')
