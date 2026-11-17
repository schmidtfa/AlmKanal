import pytest

from almkanal import AlmKanal, PhysioCleaner


@pytest.mark.filterwarnings(
    'ignore:Resampling of the stim channels caused event information to become unreliable:RuntimeWarning'
)
def test_bio(gen_mne_data_raw):
    raw, _ = gen_mne_data_raw
    ak_physio = AlmKanal(
        steps=[
            PhysioCleaner(eog='EOG 061'),
        ]
    )

    ak_physio.run(raw)
