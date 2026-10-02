from __future__ import annotations

import json

import pytest

from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package


@pytest.fixture(scope='module', autouse=True)
def _load_specs() -> None:
    load_stepspec_package('almkanal.report.stepspecs')


def test_filter_settings_pick_expected_keys() -> None:
    reg = get_registry()
    if 'Filter' not in reg:
        pytest.skip('Filter StepSpec not registered')
    info = {
        'l_freq': 0.1,
        'h_freq': 40.0,
        'method': 'fir',
        'phase': 'zero',
        'fir_window': 'hamming',
        'fir_design': 'firwin',
        'pad': 'reflect_limited',
        'l_trans_bandwidth': 'auto',
        'h_trans_bandwidth': 'auto',
        'filter_length': 'auto',
        'skip_by_annotation': ['edge'],
        'EXTRA_FIELD': 'SHOULD_BE_DROPPED',
    }
    out = reg['Filter'].settings_fn(info)
    assert 'l_freq' in out and 'h_freq' in out and 'EXTRA_FIELD' not in out
    json.dumps(out)


def test_events_and_epochs_minimal() -> None:
    reg = get_registry()
    if 'Events' not in reg or 'Epochs' not in reg:
        pytest.skip('Events/Epochs StepSpecs not registered')

    events = {
        'event_id': {'A': 1, 'B': 2},
        'stim_channel': 'STI101',
        'output': 'onset',
        'consecutive': 'increasing',
        'min_duration': 0.001,
        'shortest_event': 2,
        'mask': None,
        'uint_cast': False,
        'mask_type': 'and',
        'initial_event': False,
    }
    epochs = {
        'event_id': {'A': 1, 'B': 2},
        'tmin': -0.2,
        'tmax': 0.5,
        'baseline': [None, 0.0],
        'picks': None,
        'reject': {'mag': 4e-12},
        'flat': None,
        'proj': True,
        'reject_tmin': None,
        'reject_tmax': None,
        'detrend': None,
        'reject_by_annotation': True,
        'on_missing': 'raise',
        'event_repeated': 'error',
        'preload': True,
    }

    ev_out = reg['Events'].settings_fn(events)
    ep_out = reg['Epochs'].settings_fn(epochs)

    assert ev_out['stim_channel'] == 'STI101'
    assert ev_out['output'] == 'onset'
    assert ev_out['consecutive'] == 'increasing'
    assert ev_out['min_duration'] == 0.001
    assert 'event_id' not in ev_out

    assert ep_out['tmin'] == -0.2
    assert ep_out['tmax'] == 0.5
    assert ep_out['baseline'] == [None, 0.0]
    assert ep_out['reject'] == {'mag': 4e-12}
    assert 'event_id' not in ep_out
    assert 'preload' not in ep_out

    json.dumps(ev_out)
    json.dumps(ep_out)
