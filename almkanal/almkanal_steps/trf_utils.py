from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, List
from pathlib import Path
import numpy as np
import mne
import pandas as pd

import mne
from attrs import define

from almkanal import AlmKanalStep
from almkanal.stim_utils.audio_utils import prepare_audio


Spans = Dict[str, Tuple[int, Optional[int]]]

# def build_trf_epochs(
#     raw: mne.io.BaseRaw,
#     spans_by_label: Spans,                           # raw-sample units; can be absolute (we correct by first_samp)
#     base_audio_path: str | Path,                     # root; WAV is base / <label>.wav
#     *,
#     feature: str = "envelope",
#     audio_cutoff_hz: float = 80.0,
#     hw_delay_s: float = -0.0165,                     # + means audio lags MEG
#     epoch_len_s: float = 5.0,
#     wav_ext: str = ".wav",
# ) -> mne.Epochs:
#     """Minimal, one-pass builder: crop by spans (or audio length when off=None), then fixed-length epoch."""
#     sfreq = float(raw.info["sfreq"])
#     first = int(raw.first_samp)
#     base_audio_path = Path(base_audio_path)

#     # label -> int for epochs.event_id
#     label_to_code: Dict[str, int] = {}
#     next_code = 1
#     def code_for(lbl: str) -> int:
#         nonlocal next_code
#         if lbl not in label_to_code:
#             label_to_code[lbl] = next_code
#             next_code += 1
#         return label_to_code[lbl]

#     epochs_list: List[mne.Epochs] = []

#     for label, (on_samp, off_samp) in spans_by_label.items():
#         # 1) onset/len (seconds), auto-correct for first_samp
#         on_corr = int(on_samp) - first
#         if on_corr < 0:
#             continue
#         if off_samp is not None and off_samp > on_samp:
#             seg_len_s = (int(off_samp) - int(on_samp)) / sfreq
#         else:
#             # Need audio duration; load once via prepare_audio at MEG fs
#             wav = (base_audio_path / label).with_suffix(wav_ext)
#             if not wav.exists():
#                 raise FileNotFoundError(f"Audio missing for {label}: {wav}")
#             audio_data, _t, audio_names, fs_aud = prepare_audio(
#                 audio_path=str(wav),
#                 feature=feature,
#                 target_fs=sfreq,
#                 cutoff_hz=audio_cutoff_hz,
#             )

#             seg_len_s = audio_data.shape[1] / sfreq

#         t_on  = (on_corr / sfreq) + hw_delay_s
#         t_off = t_on + seg_len_s
#         if seg_len_s <= 0:
#             continue

#         # crop segment and add audio
#         seg = raw.copy().crop(tmin=t_on, tmax=t_off, include_tmax=False)

#         info_aud = mne.create_info(ch_names=audio_names, 
#                                    sfreq=fs_aud, 
#                                    ch_types=["misc"]*len(audio_names)
#             )
#         aud_raw = mne.io.RawArray(audio_data, info_aud)
#         seg.add_channels([aud_raw], force_update_info=True)

#         # make fixed-length epochs
#         label_code = code_for(label)
#         ep = mne.make_fixed_length_epochs(
#             seg, 
#             duration=epoch_len_s, 
#             overlap=0,
#             preload=True, 
#             id=label_code, 
#             reject_by_annotation=True
#         )

#         ep.event_id = {label: label_code}
#         ep.metadata = pd.DataFrame({
#             "label": [label] * len(ep),
#             "segment_on_s": [t_on] * len(ep),
#             "segment_off_s": [t_off] * len(ep),
#         })
#         epochs_list.append(ep)

#     if not epochs_list:
#         raise RuntimeError("No epochs produced. Check spans/audio paths/durations.")

#     epochs_all = mne.concatenate_epochs(epochs_list, on_mismatch="warn")
#     epochs_all.event_id = label_to_code
#     return epochs_all


def build_trf_epochs(
    raw: mne.io.BaseRaw,
    spans_by_label: Spans,                 # raw-sample units; absolute ok (we correct by first_samp)
    base_audio_path: str | Path,           # root; WAV is base / <label>.wav
    *,
    feature: str = "envelope",
    audio_cutoff_hz: float = 80.0,
    hw_delay_s: float = -0.0165,           # + means audio lags MEG
    epoch_len_s: float = 5.0,
    overlap: float = 0.0,                  # NEW: pass straight to MNE, used for per-epoch times
    wav_ext: str = ".wav",
) -> mne.Epochs:
    """Crop by spans (or audio length when off=None), attach audio as misc, then fixed-length epoch.
       Adds epoch-specific t_on/t_off to metadata.
    """
    sfreq = float(raw.info["sfreq"])
    first = int(raw.first_samp)
    base_audio_path = Path(base_audio_path)

    # label -> int for epochs.event_id
    label_to_code: Dict[str, int] = {}
    next_code = 1
    def code_for(lbl: str) -> int:
        nonlocal next_code
        if lbl not in label_to_code:
            label_to_code[lbl] = next_code
            next_code += 1
        return label_to_code[lbl]

    # stride for per-epoch timing
    if not (0.0 <= overlap < epoch_len_s):
        raise ValueError("overlap must satisfy 0.0 <= overlap < epoch_len_s")
    stride = epoch_len_s - overlap

    epochs_list: List[mne.Epochs] = []

    for label, (on_samp, off_samp) in spans_by_label.items():
        # always load audio (we attach it as misc)
        wav = (base_audio_path / label).with_suffix(wav_ext)
        if not wav.exists():
            raise FileNotFoundError(f"Audio missing for {label}: {wav}")

        audio_data, _t, audio_names, fs_aud = prepare_audio(
            audio_path=str(wav),
            feature=feature,
            target_fs=sfreq,
            cutoff_hz=audio_cutoff_hz,
        )
        if audio_data.ndim != 2:
            raise ValueError("prepare_audio must return (n_features, n_samples) as its first output.")
        if not np.isclose(fs_aud, sfreq):
            raise ValueError(f"prepare_audio must return data at raw.sfreq; got {fs_aud} vs {sfreq}")
        if not audio_names:
            audio_names = [f"aud{i}" for i in range(audio_data.shape[0])]

        # segment length (seconds), correcting for first_samp
        on_corr = int(on_samp) - first
        if on_corr < 0:
            continue

        if off_samp is not None and off_samp > on_samp:
            seg_len_s = (int(off_samp) - int(on_samp)) / sfreq
            # trim audio to span length
            keep = max(0, min(audio_data.shape[1], int(round(seg_len_s * sfreq))))
            audio_data = audio_data[:, :keep]
        else:
            seg_len_s = audio_data.shape[1] / sfreq  # fall back to audio length

        if seg_len_s <= 0:
            continue

        t_on  = (on_corr / sfreq) + hw_delay_s
        t_off = t_on + seg_len_s

        # crop segment and add audio as misc
        seg = raw.copy().crop(tmin=t_on, tmax=t_off, include_tmax=False)
        info_aud = mne.create_info(
            ch_names=audio_names, sfreq=float(fs_aud),
            ch_types=["misc"] * len(audio_names),
        )
        aud_raw = mne.io.RawArray(audio_data, info_aud)
        seg.add_channels([aud_raw], force_update_info=True)

        # fixed-length epochs
        label_code = code_for(label)
        ep = mne.make_fixed_length_epochs(
            seg,
            duration=epoch_len_s,
            overlap=overlap,
            preload=True,
            id=label_code,
            reject_by_annotation=True,
        )
        if not len(ep):
            continue

        # --- epoch-specific timing metadata ---
        n_ep = len(ep)
        # starts are t_on + i*stride; ends add epoch_len_s
        epoch_on_s  = t_on + np.arange(n_ep, dtype=float) * stride
        epoch_off_s = epoch_on_s + epoch_len_s
        # Cap any tiny floating overrun at segment end (numerical safety)
        epoch_off_s = np.minimum(epoch_off_s, t_off)

        ep.event_id = {label: label_code}
        ep.metadata = pd.DataFrame({
            "label": [label] * n_ep,
            "segment_on_s": [t_on] * n_ep,
            "segment_off_s": [t_off] * n_ep,
            "t_on": epoch_on_s,
            "t_off": epoch_off_s,
            "epoch_index_in_segment": np.arange(n_ep, dtype=int),
        })

        epochs_list.append(ep)

    if not epochs_list:
        raise RuntimeError("No epochs produced. Check spans/audio paths/durations.")

    epochs_all = mne.concatenate_epochs(epochs_list, on_mismatch="warn")
    epochs_all.event_id = label_to_code
    return epochs_all


@define
class EpochTRF(AlmKanalStep):
    gen_spans_by_label: Callable                         # raw-sample units; can be absolute (we correct by first_samp)
    base_audio_path: str | Path                     # root; WAV is base / <label>.wav
    feature: str = "envelope"
    audio_cutoff_hz: float = 80.0
    hw_delay_s: float = -0.0165                    # + means audio lags MEG
    epoch_len_s: float = 5.0

    must_be_before: tuple = ()
    must_be_after: tuple = ()

    def run(self, data: mne.io.BaseRaw, info: dict) -> dict:


        spans_by_label = self.gen_spans_by_label(data)

        data = build_trf_epochs(
                raw=data,
                spans_by_label=spans_by_label,                        
                base_audio_path=self.base_audio_path,                     
                feature=self.feature,
                audio_cutoff_hz = self.audio_cutoff_hz,
                hw_delay_s = self.hw_delay_s,                     # + means audio lags MEG
                epoch_len_s = self.epoch_len_s,
                wav_ext = ".wav")

        return {
            'data': data,
            'TRF_info': {
                'spans_by_label': spans_by_label,                                       
                'feature': self.feature,
                'audio_cutoff_hz': self.audio_cutoff_hz,
                'hw_delay_s': self.hw_delay_s,                    
                'epoch_len_s': self.epoch_len_s,
            },
        }

    def reports(self, data: mne.BaseEpochs, report: mne.Report, info: dict) -> None:

        report.add_epochs(data, title='Epoched (TRF)')
