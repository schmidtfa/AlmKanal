# Basic Resting state analysis pipeline
from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal


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

        ak = AlmKanal(raw=raw)
        # do raw preproc
        ak.do_maxwell()
        # % filter
        ak.raw.filter(l_freq=hp, h_freq=lp)
        # % run_ica
        ak.do_ica()
        # % do fwd model
        ak.do_fwd_model(subject_id=subject_id, subjects_dir=subjects_dir, redo_hdm=True)
        # % do epochs
        # TODO: This is a placeholder built the proper one from here
        ak.do_events()
        event_dict = {
            'Auditory/Left': 1,
            'Auditory/Right': 2,
            'Visual/Left': 3,
            'Visual/Right': 4,
        }

        ak.do_epochs(event_id=event_dict)

        ak.do_spatial_filters(empty_room_path=empty_room_path,)

        # % go 2 source
        stc = ak.do_src(
            subject_id=subject_id,
            subjects_dir=subjects_dir,
            return_parc=True,
            empty_room_path=empty_room_path,
        )

        joblib.dump(stc, self.full_output_path)
