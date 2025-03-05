from almkanal import AlmKanal, PhysioCleaner


def test_bio(gen_mne_data_raw):

    raw, _ = gen_mne_data_raw
    ak_physio = AlmKanal( steps=[
        PhysioCleaner(eog='EOG 061'),
    ])

    ak_physio.run(raw)
