import numpy as np


def check_raw_epoch(self) -> None:  # type: ignore
    # TODO: this should run whenever the data object is used to make sure that only epoched or raw data is there

    """We want to make sure that we either have epoched or raw data supplied to the object."""

    if self._data_check_disabled:
        return

    if np.logical_and(self.raw is not None, self.epoched is None):
        self.info.raw = True
        self.info.epoched = False
        self.raw.fix_mag_coil_types()  # https://mne.tools/stable/generated/mne.channels.fix_mag_coil_types.html

    elif np.logical_and(self.raw is None, self.epoched is not None):
        self.info.raw = False
        self.info.epoched = True

    elif np.logical_and(self.raw is None, self.epoched is None) or np.logical_and(
        self.raw is not None, self.epoched is not None
    ):
        raise ValueError('This pipeline needs to be intialized using either an `mne.io.Raw` or `mne.Epochs` object.')
