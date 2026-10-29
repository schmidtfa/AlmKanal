import numpy as np
import pytest
from scipy.io import wavfile
from almkanal.stim_utils.audio_utils import (
    resample_poly_exact, prepare_audio
)

@pytest.fixture
def sine_wave():
    sfreq = 44_100
    duration = 1.0

    times = np.arange(
        int(sfreq * duration)
    ) / sfreq

    audio = np.sin(
        2 * np.pi * 440 * times
    )

    return audio, sfreq



def test_resample_poly_exact_smoke(
    sine_wave,
):
    audio, sfreq = sine_wave

    target_sfreq = 100

    resampled = resample_poly_exact(
        audio,
        sfreq,
        target_sfreq,
    )

    assert resampled.ndim == 1

    assert len(resampled) == pytest.approx(
        target_sfreq,
        abs=1,
    )

    assert np.all(
        np.isfinite(resampled)
    )


@pytest.mark.parametrize(
    'target_sfreq',
    [
        100,
        200,
        512,
    ],
)
def test_resample_poly_exact_preserves_duration(
    sine_wave,
    target_sfreq,
):
    audio, sfreq = sine_wave

    resampled = resample_poly_exact(
        audio,
        sfreq,
        target_sfreq,
    )

    input_duration = (
        len(audio) / sfreq
    )

    output_duration = (
        len(resampled)
        / target_sfreq
    )

    assert output_duration == pytest.approx(
        input_duration,
        abs=1 / target_sfreq,
    )

    assert np.all(
        np.isfinite(resampled)
    )


def test_resample_poly_exact_not_empty(
    sine_wave,
):
    audio, sfreq = sine_wave

    resampled = resample_poly_exact(
        audio,
        sfreq,
        200,
    )

    assert len(resampled) > 0

    assert np.max(
        np.abs(resampled)
    ) > 0


def test_resample_poly_exact_zero_signal():
    audio = np.zeros(44_100)

    resampled = resample_poly_exact(
        audio,
        44_100,
        100,
    )

    assert len(resampled) == 100

    np.testing.assert_allclose(
        resampled,
        0,
        atol=1e-15,
    )


def test_resample_poly_exact_multichannel():
    sfreq = 1000

    times = np.arange(sfreq) / sfreq

    audio = np.vstack([
        np.sin(2 * np.pi * 10 * times),
        np.sin(2 * np.pi * 20 * times),
    ])

    resampled = resample_poly_exact(
        audio,
        sfreq,
        100,
    )

    assert resampled.shape[0] == 2
    assert resampled.shape[1] == 100

    assert np.all(
        np.isfinite(resampled)
    )


@pytest.fixture
def prep_audio_smoke(tmp_path):
    sfreq = 44_100
    duration = 1.0

    times = np.arange(
        int(sfreq * duration)
    ) / sfreq

    audio = 0.5 * np.sin(
        2 * np.pi * 220 * times
    )

    audio_file = tmp_path / 'sine.wav'

    wavfile.write(
        audio_file,
        sfreq,
        audio.astype(np.float32),
    )

    return audio_file, sfreq

@pytest.mark.parametrize(
    'feature',
    [
        'envelope',
        'mel',
        'flux',
        'mel_onsets',
        'pitch',
        'pitch_voicing',
    ],
)
def test_prepare_audio_smoke(prep_audio_smoke,
                             feature):
    audio, sfreq = prep_audio_smoke

    result = prepare_audio(
        audio,
        feature=feature,
    )

    assert result is not None