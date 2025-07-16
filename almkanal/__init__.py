"""AlmKanal Core Functions."""

from almkanal.__version__ import __version__
from almkanal.almkanal import AlmKanal, AlmKanalStep
from almkanal.preproc_utils.bio_utils import PhysioCleaner
from almkanal.preproc_utils.channel_utils import RANSAC, Maxwell, MultiBlockMaxwell, ReReference
from almkanal.preproc_utils.epoch_utils import Epochs
from almkanal.preproc_utils.event_utils import Events
from almkanal.preproc_utils.filter_utils import Filter, Resample
from almkanal.preproc_utils.ica_utils import ICA
from almkanal.src_utils.headmodel_utils import ForwardModel
from almkanal.src_utils.spatial_filter_utils import SpatialFilter
from almkanal.src_utils.src_recon_utils import SourceReconstruction

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
    'ForwardModel',
    'SpatialFilter',
    'SourceReconstruction',
    '__version__',
]
