"""Blink detection and blink-related utilities."""

from __future__ import annotations

import logging
from typing import Iterable

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def find_blink_samples(
    signal: np.ndarray,
) -> list[tuple[int, int]]:
    """Find contiguous regions where a binary signal equals one.

    Parameters
    ----------
    signal
        One-dimensional binary signal.

    Returns
    -------
    segments
        List of ``(start_sample, end_sample)`` tuples. ``end_sample`` is
        exclusive.
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    binary = signal.astype(bool)

    if not np.any(binary):
        return []

    padded = np.concatenate(
        [[False], binary, [False]]
    ).astype(int)

    transitions = np.diff(padded)

    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    return list(zip(starts, ends))


def blinks_to_annotations(
    raw: mne.io.BaseRaw,
    blink_channel: str,
    affected_channels: Iterable[str],
    *,
    description: str = "BAD_blink",
) -> mne.Annotations:
    """Convert a binary blink channel into MNE annotations.

    Parameters
    ----------
    raw
        MNE Raw object containing the blink channel.
    blink_channel
        Name of the binary blink channel.
    affected_channels
        Eye-tracking channels affected by the blink.
    description
        Annotation description.

    Returns
    -------
    annotations
        Blink annotations.
    """
    if blink_channel not in raw.ch_names:
        raise ValueError(
            f"Blink channel {blink_channel!r} was not found."
        )

    sfreq = raw.info["sfreq"]
    blink_data = raw.get_data(picks=[blink_channel])[0]

    segments = find_blink_samples(blink_data)

    if not segments:
        return mne.Annotations(
            onset=[],
            duration=[],
            description=[],
        )

    starts, ends = zip(*segments)

    return mne.Annotations(
        onset=np.asarray(starts) / sfreq,
        duration=(np.asarray(ends) - np.asarray(starts)) / sfreq,
        description=[description] * len(segments),
        ch_names=[list(affected_channels)] * len(segments),
    )


def call_blink_annotations(
    raw: mne.io.BaseRaw,
    blink_map: dict[str, Iterable[str]],
) -> mne.Annotations:
    """Create blink annotations for all channels in a blink map.

    Parameters
    ----------
    raw
        Eye-tracking Raw object.
    blink_map
        Mapping from blink-channel names to affected channels.

    Returns
    -------
    annotations
        Combined MNE annotations.
    """
    annotations = mne.Annotations(
        onset=[],
        duration=[],
        description=[],
    )

    for blink_channel, affected_channels in blink_map.items():
        logger.debug(
            "Processing blink channel %s.",
            blink_channel,
        )

        current = blinks_to_annotations(
            raw,
            blink_channel,
            affected_channels,
        )

        annotations += current

    return annotations


def vpixx_default_blinkmap() -> dict[str, tuple[str, ...]]:
    """Return the default blink-channel mapping for VPixx recordings."""
    return {
        "Left Eye Blink": (
            "Left Eye x",
            "Left Eye y",
            "Left Eye Raw x",
            "Left Eye Raw y",
            "Left Eye Pupil Diameter",
        ),
        "Right Eye Blink": (
            "Right Eye x",
            "Right Eye y",
            "Right Eye Raw x",
            "Right Eye Raw y",
            "Right Eye Pupil Diameter",
        ),
    }


def _eye_from_channel_names(
    ch_names: Iterable[str],
) -> tuple[bool, bool]:
    """Determine whether channel names refer to left/right eyes."""
    names = [name.lower() for name in ch_names]

    return (
        any("left" in name for name in names),
        any("right" in name for name in names),
    )


def blink_stats_from_annotations(
    annotations: mne.Annotations,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Extract blink statistics from MNE annotations.

    Parameters
    ----------
    annotations
        MNE annotations containing ``BAD_blink`` or ``blink`` events.

    Returns
    -------
    stats
        Dictionary containing blink counts, durations and inter-blink
        intervals separately for the left and right eyes.
    """
    onsets: dict[str, list[float]] = {
        "left": [],
        "right": [],
    }

    durations: dict[str, list[float]] = {
        "left": [],
        "right": [],
    }

    for annotation in annotations:
        if annotation["description"] not in {
            "BAD_blink",
            "blink",
        }:
            continue

        left, right = _eye_from_channel_names(
            annotation["ch_names"]
        )

        if left:
            onsets["left"].append(float(annotation["onset"]))
            durations["left"].append(float(annotation["duration"]))

        if right:
            onsets["right"].append(float(annotation["onset"]))
            durations["right"].append(float(annotation["duration"]))

    stats: dict[str, dict[str, np.ndarray | int]] = {}

    for eye in ("left", "right"):
        onset = np.asarray(onsets[eye], dtype=float)
        duration = np.asarray(durations[eye], dtype=float)

        order = np.argsort(onset)
        onset = onset[order]
        duration = duration[order]

        ibi = np.diff(onset) if len(onset) > 1 else np.array([])

        stats[eye] = {
            "n_blinks": len(onset),
            "durations": duration,
            "ibi": ibi,
        }

    return stats


# ---------------------------------------------------------------------------
# EOG blink detection
# ---------------------------------------------------------------------------

def get_blinks_eog_infos(
    eog: np.ndarray,
    *,
    sampling_rate: float = 1000,
    threshold_percentile: float = 75,
    window_samples: int | None = None,
) -> pd.DataFrame:
    """Extract blink onset/offset information from an EOG signal.

    Parameters
    ----------
    eog
        One-dimensional EOG signal.
    sampling_rate
        Sampling frequency in Hz.
    threshold_percentile
        Percentile used to determine blink boundaries.
    window_samples
        Search window around each detected blink peak.

    Returns
    -------
    blinks
        DataFrame containing blink peak, onset, offset and duration.
    """
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError(
            "EOG blink detection requires neurokit2. "
            "Install it with `pip install neurokit2`."
        ) from exc

    eog = np.asarray(eog, dtype=float)

    if eog.ndim != 1:
        raise ValueError("eog must be one-dimensional.")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    if not 0 <= threshold_percentile <= 100:
        raise ValueError(
            "threshold_percentile must be between 0 and 100."
        )

    if window_samples is None:
        window_samples = int(sampling_rate // 2)

    eog_signals, _ = nk.eog_process(
        np.abs(eog),
        sampling_rate=sampling_rate,
    )

    blink_peaks = np.flatnonzero(
        eog_signals["EOG_Blinks"].to_numpy() == 1
    )

    clean_abs = np.abs(
        eog_signals["EOG_Clean"].to_numpy()
    )

    threshold = np.percentile(
        clean_abs,
        threshold_percentile,
    )

    onsets = []
    offsets = []

    for peak in blink_peaks:
        onset = peak
        for index in range(
            peak,
            max(-1, peak - window_samples),
            -1,
        ):
            if clean_abs[index] < threshold:
                onset = index
                break

        offset = peak
        for index in range(
            peak,
            min(len(clean_abs), peak + window_samples),
        ):
            if clean_abs[index] < threshold:
                offset = index
                break

        onsets.append(onset)
        offsets.append(offset)

    return pd.DataFrame(
        {
            "peak_samples": blink_peaks,
            "onset_samples": onsets,
            "offset_samples": offsets,
            "onset_sec": np.asarray(onsets) / sampling_rate,
            "offset_sec": np.asarray(offsets) / sampling_rate,
            "duration_sec": (
                np.asarray(offsets) - np.asarray(onsets)
            ) / sampling_rate,
        }
    )


def add_blinkvec2raw(
    raw: mne.io.BaseRaw,
    *,
    hp_freq: float = 0.1,
    lp_freq: float = 10.0,
    eoglab: list[str] | None = None,
    thresh: float = 75,
) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    """Detect EOG blinks and add a blink channel and annotations.

    Parameters
    ----------
    raw
        MNE Raw object containing EOG data.
    hp_freq
        High-pass filter frequency in Hz.
    lp_freq
        Low-pass filter frequency in Hz.
    eoglab
        EOG channel names. Defaults to ``["EOG001"]``.
    thresh
        Percentile threshold used for blink detection.

    Returns
    -------
    raw
        Modified Raw object.
    blinks_df
        DataFrame containing detected blink intervals.

    Notes
    -----
    The EOG signal is filtered on a copy of the Raw object, so the original
    EOG channels are not modified.
    """
    if eoglab is None:
        eoglab = ["EOG001"]

    missing = set(eoglab) - set(raw.ch_names)

    if missing:
        raise ValueError(
            f"EOG channels not found in Raw object: {sorted(missing)}"
        )

    if hp_freq >= lp_freq:
        raise ValueError("hp_freq must be lower than lp_freq.")

    sfreq = raw.info["sfreq"]
    n_times = raw.n_times

    eog = (
        raw.copy()
        .filter(hp_freq, lp_freq, picks=eoglab)
        .get_data(picks=eoglab)
    )

    eog_signal = (
        eog.mean(axis=0)
        if eog.shape[0] > 1
        else eog[0]
    )

    blinks_df = get_blinks_eog_infos(
        eog_signal,
        sampling_rate=sfreq,
        threshold_percentile=thresh,
    )

    blink_vec = np.zeros(
        n_times,
        dtype=np.int8,
    )

    onsets = np.clip(
        blinks_df["onset_samples"].to_numpy(dtype=int),
        0,
        n_times,
    )

    offsets = np.clip(
        blinks_df["offset_samples"].to_numpy(dtype=int),
        0,
        n_times,
    )

    for onset, offset in zip(onsets, offsets):
        if offset > onset:
            blink_vec[onset:offset] = 1

    info = mne.create_info(
        ["BLINK"],
        sfreq=sfreq,
        ch_types=["misc"],
    )

    blink_raw = mne.io.RawArray(
        blink_vec[np.newaxis, :],
        info,
    )

    raw.add_channels(
        [blink_raw],
        force_update_info=True,
    )

    annotations = mne.Annotations(
        onset=blinks_df["onset_sec"].to_numpy(),
        duration=blinks_df["duration_sec"].to_numpy(),
        description=["blink"] * len(blinks_df),
    )

    raw.set_annotations(raw.annotations + annotations)

    return raw, blinks_df
