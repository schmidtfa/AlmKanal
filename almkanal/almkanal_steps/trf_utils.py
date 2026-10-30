"""Compatibility imports for stimulus epoching and saved pipeline objects."""

from almkanal.almkanal_steps.stimulus_utils import (
    EpochStimuls,
    EpochTRF,
    StimulusSpanSpec,
    TRFSpanSpec,
    build_stimulus_epochs,
    build_trf_epochs,
)

__all__ = [
    'EpochStimuls',
    'EpochTRF',
    'StimulusSpanSpec',
    'TRFSpanSpec',
    'build_stimulus_epochs',
    'build_trf_epochs',
]
