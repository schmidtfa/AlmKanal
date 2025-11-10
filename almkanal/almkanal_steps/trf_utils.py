from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import attrs
import mne
import numpy as np
import pandas as pd
from attrs import define, field

from almkanal import AlmKanalStep
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
    """

    spans_by_label: Spans = field(validator=_validate_spans)
    metadata_by_label: MetaMap = field(factory=dict, validator=_validate_meta)

    def labels(self) -> list:
        # preserve insertion order from user’s dict (Python 3.7+)
        return list(self.spans_by_label.keys())


def build_trf_epochs(  # noqa: C901, PLR0915
    raw: mne.io.BaseRaw,
    spec: TRFSpanSpec,  # <<< use the attrs object
    base_audio_path: str | Path,  # root; WAV is base / <label>.wav
    *,
    feature: str = 'envelope',
    audio_cutoff_hz: float = 80.0,
    hw_delay_s: float = -0.0165,
    epoch_len_s: float = 5.0,
    wav_ext: str = '.wav',
) -> mne.Epochs:
    sfreq = float(raw.info['sfreq'])
    first = int(raw.first_samp)
    base_audio_path = Path(base_audio_path)

    # label -> int for epochs.event_id
    label_to_code: dict[str, int] = {}
    next_code = 1

    def code_for(lbl: str) -> int:
        nonlocal next_code
        if lbl not in label_to_code:
            label_to_code[lbl] = next_code
            next_code += 1
        return label_to_code[lbl]

    stride = epoch_len_s

    epochs_list: list[mne.Epochs] = []

    for label in spec.labels():
        on_samp, off_samp = spec.spans_by_label[label]

        # always load audio (we attach it as misc and/or need its duration)
        wav = (base_audio_path / label).with_suffix(wav_ext)
        if not wav.exists():
            raise FileNotFoundError(f'Audio missing for {label}: {wav}')

        audio_data, _t, audio_names, fs_aud = prepare_audio(
            audio_path=str(wav),
            feature=feature,
            target_fs=sfreq,
            cutoff_hz=audio_cutoff_hz,
        )

        on_corr = int(on_samp) - first
        if on_corr < 0:
            # span starts before available data after prior crop -> skip safely
            continue

        if off_samp is not None:
            seg_len_s = (int(off_samp) - int(on_samp)) / sfreq
            keep = max(0, min(audio_data.shape[1], int(round(seg_len_s * sfreq))))
            audio_data = audio_data[:, :keep]
        else:
            seg_len_s = audio_data.shape[1] / sfreq

        if seg_len_s <= 0:
            continue

        t_seg_on = (on_corr / sfreq) + hw_delay_s

        # audio length in samples
        n_aud = int(audio_data.shape[1])

        # how many MEG samples are actually available starting at t_seg_on?
        # (+1 because include_tmax=True includes the last sample)
        n_meg_avail = max(0, int(np.floor((raw.times[-1] - t_seg_on) * sfreq)) + 1)

        # we can only keep the minimum of the two
        desired_n = min(n_aud, n_meg_avail)
        if desired_n <= 0:
            continue

        # trim audio to exactly the number of MEG samples we can obtain
        if n_aud != desired_n:
            audio_data = audio_data[:, :desired_n]

        #t_seg_off = t_seg_on + seg_len_s
        t_seg_off = t_seg_on + (desired_n - 1) / sfreq

        # crop segment and add audio as misc
        seg = raw.copy().crop(tmin=t_seg_on, tmax=t_seg_off, include_tmax=True)

        info_aud = mne.create_info(
            ch_names=audio_names,
            sfreq=float(fs_aud),
            ch_types=['misc'] * len(audio_names),
        )
        aud_raw = mne.io.RawArray(audio_data, info_aud)
        seg.add_channels([aud_raw], force_update_info=True)

        # fixed-length epochs
        label_code = code_for(label)
        ep = mne.make_fixed_length_epochs(
            seg,
            duration=epoch_len_s,
            overlap=0,
            preload=True,
            id=label_code,
            reject_by_annotation=True,
        )
        if not len(ep):
            continue

        # per-epoch timing
        n_ep = len(ep)
        epoch_on_s = t_seg_on + np.arange(n_ep, dtype=float) * stride
        epoch_off_s = np.minimum(epoch_on_s + epoch_len_s, t_seg_off)

        # merge per-label metadata (broadcast scalars; allow sequence length == n_ep)
        extra = spec.metadata_by_label.get(label, {})
        meta = {
            'label': [label] * n_ep,
            'segment_on_s': [t_seg_on] * n_ep,
            'segment_off_s': [t_seg_off] * n_ep,
            't_on': epoch_on_s,
            't_off': epoch_off_s,
            'epoch_index_in_segment': np.arange(n_ep, dtype=int),
        }
        for k, v in extra.items():
            if hasattr(v, '__len__') and not isinstance(v, str | bytes) and len(v) == n_ep:
                meta[k] = list(v)
            else:
                meta[k] = [v] * n_ep

        ep.event_id = {label: label_code}
        ep.metadata = pd.DataFrame(meta)
        epochs_list.append(ep)

    if not epochs_list:
        raise RuntimeError('No epochs produced. Check spans/audio paths/durations.')

    epochs_all = mne.concatenate_epochs(epochs_list, on_mismatch='warn')
    epochs_all.event_id = label_to_code
    return epochs_all


@define
class EpochTRF(AlmKanalStep):
    gen_span_spec: Callable  # raw-sample units; can be absolute (we correct by first_samp)
    base_audio_path: str | Path  # root; WAV is base / <label>.wav
    feature: str = 'envelope'
    audio_cutoff_hz: float = 80.0
    hw_delay_s: float = -0.0165  # + means audio lags MEG
    epoch_len_s: float = 5.0

    must_be_before: tuple = ()
    must_be_after: tuple = ()

    def run(self, data: mne.io.BaseRaw, info: dict) -> dict:
        # spans_by_label = self.gen_spans_by_label(data)

        spec: TRFSpanSpec = self.gen_span_spec(data)  # <- attrs object

        data = build_trf_epochs(
            raw=data,
            spec=spec,
            base_audio_path=self.base_audio_path,
            feature=self.feature,
            audio_cutoff_hz=self.audio_cutoff_hz,
            hw_delay_s=self.hw_delay_s,
            epoch_len_s=self.epoch_len_s,
            wav_ext='.wav',
        )

        return {
            'data': data,
            'TRF_info': {
                'spans_by_label': spec.spans_by_label,
                'feature': self.feature,
                'audio_cutoff_hz': self.audio_cutoff_hz,
                'hw_delay_s': self.hw_delay_s,
                'epoch_len_s': self.epoch_len_s,
            },
        }

    def reports(self, data: mne.BaseEpochs, report: mne.Report, info: dict) -> None:
        report.add_epochs(data, title='Epoched (TRF)')
