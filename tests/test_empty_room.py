import pytest
import os
from datetime import datetime, timezone
from pathlib import Path
import mne

# Adjust the import path if needed
from almkanal.src_utils.spatial_filter_utils import get_nearest_empty_room

@pytest.fixture
def setup_empty_room_dir(tmp_path):
    """
    Creates three date-labeled subfolders (YYMMDD). Two contain only 'supine',
    and one contains only 'empty_room_68.fif' (the target).
    """
    # 1) 240101 => supine file only
    d1 = tmp_path / "240101"
    d1.mkdir()
    (d1 / "empty_room_supine.fif").touch()

    # 2) 240110 => 'empty_room_68.fif' only (our expected pickup)
    d2 = tmp_path / "240110"
    d2.mkdir()
    (d2 / "empty_room_68.fif").touch()

    # 3) 240115 => supine file only
    d3 = tmp_path / "240115"
    d3.mkdir()
    (d3 / "empty_room_supine.fif").touch()

    return tmp_path


@pytest.fixture
def mock_info():
    """
    Creates a real MNE Info object with a UTC-aware datetime for meas_date.
    MNE enforces that if meas_date is a datetime, it must be UTC.
    """
    info = mne.create_info(ch_names=["MEG 001"], sfreq=1000, ch_types=["mag"])
    info.set_meas_date(datetime(2024, 1, 11, tzinfo=timezone.utc))
    return info


def test_get_nearest_empty_room(setup_empty_room_dir, mock_info):
    """
    Ensures `get_nearest_empty_room()` picks the 'empty_room_68.fif' file
    in '240110', which is closest to 2024-01-11 and doesn't get skipped.
    """
    # For debugging, see what was created
    print("Test directories:", os.listdir(setup_empty_room_dir))
    for date_dir in os.listdir(setup_empty_room_dir):
        print(f"  {date_dir} -> {os.listdir(setup_empty_room_dir / date_dir)}")

    # Call your original function
    result_path = get_nearest_empty_room(mock_info, str(setup_empty_room_dir))

    # We expect the function to pick folder "240110" => file "empty_room_68.fif"
    expected = setup_empty_room_dir / "240110" / "empty_room_68.fif"
    assert result_path == expected, f"Expected {expected}, but got {result_path}"
