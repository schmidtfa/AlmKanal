"""AlmKanal Core Functions."""

from almkanal.__version__ import __version__
from almkanal.almkanal import AlmKanal, AlmKanalStep
from almkanal.preproc_utils.maxwell_utils import Maxwell
from almkanal.preproc_utils.ica_utils import ICA
from almkanal.src_utils.headmodel_utils import ForwardModel
from almkanal.src_utils.spatial_filter_utils import SpatialFilter

from almkanal.almkanal_functions import (
                                         do_bio_process,
                                         do_src)

__all__ = ['AlmKanal',
           'AlmKanalStep', 
           'Maxwell',
           'ICA',
           'ForwardModel',
           'SpatialFilter',
           'do_bio_process',
           'do_src',
           '__version__']
