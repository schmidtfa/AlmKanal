from almkanal.info import AlmKanalInfo, StepInfo

def test_info_retains_repeated_steps():
    info = AlmKanalInfo()

    info.add(StepInfo('Filter', {'l_freq': 1}))
    info.add(StepInfo('ICA', {'n_components': 20}))
    info.add(StepInfo('Filter', {'l_freq': 2}))

    assert len(info.processing_history) == 3
    assert info.get_step('Filter').info['l_freq'] == 2
    assert info.get_step('Filter', occurrence=0).info['l_freq'] == 1
    assert len(info.get_steps('Filter')) == 2