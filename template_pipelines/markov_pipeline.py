# Basic Resting state analysis pipeline
from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import (
    
    AlmKanal,
    Maxwell,
    Filter,
    ICA,
    Events,
    Epochs,
    ForwardModel,
    SpatialFilter,
    SourceReconstruction,
)


class RestingPipe(Job):
    job_data_folder = 'data_meg'

    def run(
        self,
        subject_id: str,
        data_path: str,
        subjects_dir: str,
        empty_room_path: str,
        lp: float = 100,
        hp: float = 0.1,
    ) -> None:
        full_path = Path(data_path) / subject_id + '_resting.fif'
        raw = mne.io.read_raw(full_path, preload=True)


        pick_dict = {
                    'meg': True,
                    'eog': True,
                    'ecg': True,
                    'eeg': False,
                    }
        event_dict = {
            'Auditory/Left': 1,
            'Auditory/Right': 2,
            'Visual/Left': 3,
            'Visual/Right': 4,
        }

        ak = AlmKanal(
                    pick_params=pick_dict,
                    steps=[
                        Maxwell(),
                        Filter(highpass=hp, lowpass=lp),
                        ICA(
                            train=True,
                            eog=True,
                            ecg=True,
                            emg=True,
                            resample_freq=200,
                        ),
                        Events(),
                        Epochs(event_id=event_dict),
                        ForwardModel(subject_id=subject_id, subjects_dir=subjects_dir, redo_hdm=True),
                        SpatialFilter(),
                        SourceReconstruction(subject_id=subject_id, subjects_dir=subjects_dir, return_parc=True,),
                    ],
        )
        stc = ak.run(raw)

        joblib.dump(stc, self.full_output_path)
