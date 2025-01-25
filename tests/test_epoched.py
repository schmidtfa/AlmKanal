import mne
from almkanal.almkanal import AlmKanal


def test_src(gen_mne_data_epochs): 

    ak = AlmKanal(epoched=gen_mne_data_epochs)

    data_path = mne.datasets.sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    fwd_fname = meg_path / 'sample_audvis-meg-vol-7-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    ak.pick_dict['meg'] = 'mag'
    ak.fwd = fwd
    ak.do_spatial_filters()
    ak.do_src()