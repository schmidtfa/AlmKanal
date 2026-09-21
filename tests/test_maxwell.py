import mne
from almkanal.almkanal_steps.channel_utils import Maxwell
from almkanal import AlmKanal



def test_maxwell(gen_mne_data_raw):

    raw, data_path = gen_mne_data_raw

    ak = AlmKanal(steps=[Maxwell()])
    ak.run(raw)


def test_maxwell_preserves_existing_bads(
    gen_mne_data_raw,
    monkeypatch,
):
    raw, _ = gen_mne_data_raw
    raw = raw.copy()

    raw.info['bads'] = ['MEG 2443']

    def fake_find_bads(*args, **kwargs):
        return ['MEG 0111'], ['MEG 0121']

    captured = {}

    def fake_maxwell(raw, **kwargs):
        captured['bads'] = raw.info['bads'].copy()
        return raw

    monkeypatch.setattr(
        mne.preprocessing,
        'find_bad_channels_maxwell',
        fake_find_bads,
    )

    monkeypatch.setattr(
        mne.preprocessing,
        'maxwell_filter',
        fake_maxwell,
    )

    step = Maxwell()
    result = step.run(
        raw,
        info={},
    )

    assert captured['bads'] == [
        'MEG 2443',
        'MEG 0111',
        'MEG 0121',
    ]

    assert result['data'] is raw