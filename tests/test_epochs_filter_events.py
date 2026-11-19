from almkanal.info import AlmKanalInfo
from almkanal import Epochs

import numpy as np
import pandas as pd
import mne
import pytest


def make_test_data():
    """Create synthetic raw data with 1,100 events.

    17 events have code 128 (auditory).
    1,083 events have code 256 (other).
    """
    sfreq = 1000
    n_samples = 120_000

    info = mne.create_info(
        ch_names=["MEG001"],
        sfreq=sfreq,
        ch_types=["mag"],
    )

    rng = np.random.default_rng(42)
    raw_data = rng.standard_normal((1, n_samples)) * 1e-12
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    samples = np.linspace(
        1000,
        n_samples - 1000,
        1100,
        dtype=int,
    )

    codes = np.full(1100, 256)
    codes[:17] = 128

    events = np.column_stack([
        samples,
        np.zeros(1100, dtype=int),
        codes,
    ])

    metadata = pd.DataFrame({
        "trial": np.arange(17),
        "condition": ["auditory"] * 17,
    })

    return raw, events, metadata


def create_epochs(raw, events, event_id, metadata=None, filter_events=False):
    """Reproduce the event-selection and validation logic in Epochs.run."""
    all_events = events

    if filter_events and event_id is not None:
        selected_ids = list(event_id.values())
        events = events[np.isin(events[:, 2], selected_ids)]

    if metadata is not None and len(metadata) != len(events):
        raise ValueError(
            f"Metadata contains {len(metadata)} rows, "
            f"but {len(events)} events were supplied to Epochs."
        )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0,
        tmax=0.1,
        baseline=None,
        metadata=metadata,
        preload=True,
        verbose=False,
    )

    return epochs, all_events, events


def test_filter_events_enabled():
    """Filtering should retain only the selected event code."""
    raw, events, metadata = make_test_data()

    epochs, all_events, selected_events = create_epochs(
        raw,
        events,
        event_id={"auditory": 128},
        metadata=metadata,
        filter_events=True,
    )

    assert len(all_events) == 1100
    assert len(selected_events) == 17
    assert len(epochs) == 17
    assert np.all(selected_events[:, 2] == 128)
    assert len(epochs.metadata) == 17


def test_filter_events_disabled_with_matching_metadata():
    """Without filtering, the original event array should be retained.

    Metadata must have one row per input event.
    """
    raw, events, _ = make_test_data()

    metadata = pd.DataFrame({
        "trial": np.arange(len(events)),
    })

    epochs, all_events, selected_events = create_epochs(
        raw,
        events,
        event_id={"auditory": 128},
        metadata=metadata,
        filter_events=False,
    )

    assert len(all_events) == 1100
    assert len(selected_events) == 1100

    # MNE still selects only the auditory event when event_id is supplied.
    assert len(epochs) == 17


def test_filter_events_disabled_with_mismatched_metadata():
    """The original metadata mismatch should raise a clear error."""
    raw, events, metadata = make_test_data()

    with pytest.raises(
        ValueError,
        match="Metadata contains 17 rows",
    ):
        create_epochs(
            raw,
            events,
            event_id={"auditory": 128},
            metadata=metadata,
            filter_events=False,
        )


def test_filter_events_enabled_with_multiple_event_ids():
    """Filtering should retain all event codes specified in event_id."""
    raw, events, _ = make_test_data()

    # Change five of the non-auditory events to code 512.
    events = events.copy()
    events[17:22, 2] = 512

    metadata = pd.DataFrame({
        "trial": np.arange(22),
    })

    epochs, _, selected_events = create_epochs(
        raw,
        events,
        event_id={
            "auditory": 128,
            "visual": 512,
        },
        metadata=metadata,
        filter_events=True,
    )

    assert len(selected_events) == 22
    assert len(epochs) == 22
    assert set(selected_events[:, 2]) == {128, 512}


def test_filter_events_enabled_without_event_id():
    """Without event_id, filtering should not change the event array."""
    raw, events, _ = make_test_data()

    metadata = pd.DataFrame({
        "trial": np.arange(len(events)),
    })

    epochs, all_events, selected_events = create_epochs(
        raw,
        events,
        event_id=None,
        metadata=metadata,
        filter_events=True,
    )

    assert len(all_events) == 1100
    assert len(selected_events) == 1100
    assert len(epochs) == 1100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_almkanal_epochs_filters_events_with_metadata():
    """Test the actual AlmKanal Epochs step with event filtering enabled."""
    raw, events, metadata = make_test_data()

    step = Epochs(
        events=events,
        event_id={"auditory": 128},
        metadata=metadata,
        filter_events=True,
    )

    info = AlmKanalInfo()
    result = step.run(raw, info)

    epochs = result["data"]
    epochs_info = result["epochs_info"]

    # Only the 17 auditory events should be used.
    assert len(epochs) == 17
    assert len(epochs.metadata) == 17

    # Original and selected events should both be preserved.
    assert len(epochs_info["events"]) == 1100
    assert len(epochs_info["selected_events"]) == 17

    assert np.all(epochs_info["selected_events"][:, 2] == 128)
    assert epochs_info["filter_events"] is True

    # Metadata should remain correctly associated with the epochs.
    pd.testing.assert_frame_equal(
        epochs.metadata.reset_index(drop=True),
        metadata.reset_index(drop=True),
    )


def test_almkanal_epochs_filter_events_defaults_to_false():
    """Test that event filtering is disabled by default."""
    raw, events, _ = make_test_data()

    # Metadata has one row for every original event.
    metadata = pd.DataFrame({
        "trial": np.arange(len(events)),
    })

    step = Epochs(
        events=events,
        event_id={"auditory": 128},
        metadata=metadata,
    )

    assert step.filter_events is False

    info = AlmKanalInfo()
    result = step.run(raw, info)

    epochs = result["data"]
    epochs_info = result["epochs_info"]

    # MNE selects the auditory epochs using event_id.
    assert len(epochs) == 17

    # The Epochs step itself should not have filtered the input events.
    assert len(epochs_info["events"]) == 1100
    assert len(epochs_info["selected_events"]) == 1100

    assert epochs_info["filter_events"] is False


def test_almkanal_epochs_raises_for_mismatched_metadata():
    """Test metadata validation in the actual Epochs step."""
    raw, events, metadata = make_test_data()

    step = Epochs(
        events=events,
        event_id={"auditory": 128},
        metadata=metadata,
        filter_events=False,
    )

    info = AlmKanalInfo()

    with pytest.raises(
        ValueError,
        match="Metadata contains 17 rows",
    ):
        step.run(raw, info)


def test_almkanal_epochs_filters_multiple_event_ids():
    """Test the actual Epochs step with multiple selected event codes."""
    raw, events, _ = make_test_data()

    events = events.copy()
    events[17:22, 2] = 512

    metadata = pd.DataFrame({
        "trial": np.arange(22),
    })

    step = Epochs(
        events=events,
        event_id={
            "auditory": 128,
            "visual": 512,
        },
        metadata=metadata,
        filter_events=True,
    )

    info = AlmKanalInfo()
    result = step.run(raw, info)

    epochs = result["data"]
    epochs_info = result["epochs_info"]

    assert len(epochs) == 22
    assert len(epochs_info["events"]) == 1100
    assert len(epochs_info["selected_events"]) == 22

    assert set(epochs_info["selected_events"][:, 2]) == {128, 512}
    assert epochs_info["filter_events"] is True
