from attrs import define
import mne
import numpy as np

#all them utility functions
from utils.maxwell_utils import run_maxwell



@define
class AlmKanal:
    raw: None | mne.io.Raw
    epoched: None | mne.Epochs
    fwd: None | mne.Forward
    events: None | np.ndarray
    ica: None | mne.preprocessing.ICA
    stc: None | mne.SourceEstimate
    stim: None | np.ndarray #This is for TRF stuff

    #TODO: We need a check that we at least have either raw or epoched data

    def gen_pipeline(self):

        pipe_steps = {'maxwell': None,
                      'filter': None,
                      'ica': None,
                      'epoched': None,
                      'potato': None,
                      'trf_epochs': None,
                      'src': None,
                      'eel': None,
                      }

        if np.logical_and(self.raw is not None, self.epoched is None):
            self.raw.info['pipeline']

        elif np.logical_and(self.epoched is not None, self.raw is None):
            self.epoched.info['pipeline']

        elif np.logical_and(self.raw is None, self.epoched is None):
            raise ValueError('This pipeline needs to be intialized using either an `mne.io.Raw` or `mne.Epochs` object.')

        elif np.logical_and(self.raw is not None, self.epoched is not None):
            raise ValueError('This pipeline needs to be intialized using either an `mne.io.Raw` or `mne.Epochs` object.')
        

    def do_maxwell(self):
        #this should do maxwell filtering
        if self.raw.info['pipeline']['maxwell'] is not None:
            self.raw.info['pipeline']['maxwell'] = {'coord_frame': self.mw_coord_frame,
                                                    'destination': self.mw_destination,
                                                    'calibration_file': self.mw_calibration_file,
                                                    'cross_talk_file': self.mw_cross_talk_file,
                                                    'st_duration': self.mw_st_duration}
            
            self.raw = run_maxwell( raw=self.raw,
                                    coord_frame=self.mw_coord_frame,
                                    destination=self.mw_destination,
                                    calibration_file=self.mw_calibration_file,
                                    cross_talk_file=self.mw_cross_talk_file,
                                    st_duration=self.mw_st_duration)
        else:
            print('You already maxfiltered your data')


    def do_filter(self):
        #this should apply hp/lp filtering
        if self.raw is not None:
            self.raw.filter()

    
    def do_ica(self):
        #this should do an ica
        pass


    def do_events(self):
        #this should build events based on information stored in the raw file
        pass


    def do_epochs(self):
        #this should take raw and events and epoch the data
        pass


    def do_potato(self):
        #This should run the riemann potato on epochs
        pass


    def do_trf_epochs(self):
        #mne only allows epochs of equal length. 
        #This should become a shorthand to split the raw file in smaller raw files based on events
        pass


    def do_fwd_model(self):
        #This should generate a fwd model either based
        pass


    def do_src(self):
        #here we want to embed the logic that, if your object has been epoched we do epoched2src else raw2src
        pass


    def convert2eelbrain(self):
        #This should take the thht mixin to convert raw, epoched or stc objects into eelbrain
        pass