"""AlmKanal Core Functions."""

from almkanal.__version__ import __version__
from almkanal.almkanal import AlmKanal, AlmKanalStep
from almkanal.almkanal_steps.bio_utils import PhysioCleaner
from almkanal.almkanal_steps.channel_utils import RANSAC, Maxwell, MultiBlockMaxwell, ReReference
from almkanal.almkanal_steps.epoch_utils import Epochs
from almkanal.almkanal_steps.event_utils import Events
from almkanal.almkanal_steps.filter_utils import Filter, Resample
from almkanal.almkanal_steps.headmodel_utils import ForwardModel
from almkanal.almkanal_steps.ica_utils import ICA
from almkanal.almkanal_steps.spatial_filter_utils import SpatialFilter
from almkanal.almkanal_steps.src_recon_utils import SourceReconstruction
from almkanal.almkanal_steps.trf_utils import EpochTRF
from almkanal.report.exporting import preprocessing_report

__all__ = [
    'AlmKanal',
    'AlmKanalStep',
    'Filter',
    'Resample',
    'Maxwell',
    'MultiBlockMaxwell',
    'ReReference',
    'RANSAC',
    'ICA',
    'PhysioCleaner',
    'Events',
    'Epochs',
    'EpochTRF',
    'ForwardModel',
    'SpatialFilter',
    'SourceReconstruction',
    'preprocessing_report',
    '__version__',
]
