"""Blink detection and blink-related utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
import mne
import numpy as np

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
        raise ValueError('signal must be one-dimensional.')

    binary = signal.astype(bool)

    if not np.any(binary):
        return []

    padded = np.concatenate([[False], binary, [False]]).astype(int)

    transitions = np.diff(padded)

    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    return list(zip(starts, ends))


def blinks_to_annotations(
    raw: mne.io.BaseRaw,
    blink_channel: str,
    affected_channels: Iterable[str],
    *,
    description: str = 'BAD_blink',
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
        raise ValueError(f'Blink channel {blink_channel!r} was not found.')

    sfreq = raw.info['sfreq']
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
    blink_map: Mapping[str, Iterable[str]],
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
            'Processing blink channel %s.',
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
        'Left Eye Blink': (
            'Left Eye x',
            'Left Eye y',
            'Left Eye Raw x',
            'Left Eye Raw y',
            'Left Eye Pupil Diameter',
        ),
        'Right Eye Blink': (
            'Right Eye x',
            'Right Eye y',
            'Right Eye Raw x',
            'Right Eye Raw y',
            'Right Eye Pupil Diameter',
        ),
    }


def _eye_from_channel_names(
    ch_names: Iterable[str],
) -> tuple[bool, bool]:
    """Determine whether channel names refer to left/right eyes."""
    names = [name.lower() for name in ch_names]

    return (
        any('left' in name for name in names),
        any('right' in name for name in names),
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
        'left': [],
        'right': [],
    }

    durations: dict[str, list[float]] = {
        'left': [],
        'right': [],
    }

    for annotation in annotations:
        if annotation['description'] not in {
            'BAD_blink',
            'blink',
        }:
            continue

        left, right = _eye_from_channel_names(annotation['ch_names'])

        if left:
            onsets['left'].append(float(annotation['onset']))
            durations['left'].append(float(annotation['duration']))

        if right:
            onsets['right'].append(float(annotation['onset']))
            durations['right'].append(float(annotation['duration']))

    stats: dict[str, dict[str, np.ndarray | int]] = {}

    for eye in ('left', 'right'):
        onset = np.asarray(onsets[eye], dtype=float)
        duration = np.asarray(durations[eye], dtype=float)

        order = np.argsort(onset)
        onset = onset[order]
        duration = duration[order]

        ibi = np.diff(onset) if len(onset) > 1 else np.array([])

        stats[eye] = {
            'n_blinks': len(onset),
            'durations': duration,
            'ibi': ibi,
        }

    return stats
