# %%
from fractions import Fraction

import librosa
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def _resample_poly_exact(x: np.ndarray, fs_in: int | float, fs_out: int | float, axis: int = 1) -> np.ndarray:
    """Polyphase resample, then trim/pad to n_out = round(T * fs_out)."""
    t_in = x.shape[axis] / float(fs_in)  # exclusive end
    n_out = int(round(t_in * float(fs_out)))
    r = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    y = resample_poly(x, up=r.numerator, down=r.denominator, axis=axis)
    cur = y.shape[axis]
    if cur < n_out:  # pad at the tail to preserve t=0 alignment
        pad_shape = list(y.shape)
        pad_shape[axis] = n_out - cur
        pad = np.zeros(pad_shape, dtype=y.dtype)
        y = np.concatenate([y, pad], axis=axis)
    elif cur > n_out:
        slicer = [slice(None)] * y.ndim
        slicer[axis] = slice(0, n_out)
        y = y[tuple(slicer)]
    return y


def prepare_audio(
    audio_path: str,
    feature: str = 'envelope',  # 'envelope' | 'mel' | 'flux' | 'rectify'
    target_fs: float = 100.0,  # desired feature rate (Hz)
    cutoff_hz: float = 80.0,  # LP for envelope-like features
    n_mels: int = 32,
    fmin: float = 50.0,
    fmax: float = 8000.0,
) -> tuple[np.ndarray, np.ndarray, list, float]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # 1) lock hop_length to target_fs
    hop = max(1, int(round(sr / float(target_fs))))
    base_fs = sr / hop  # actual grid rate (might be ~target_fs but not exact)

    names = []
    if feature == 'envelope':
        # frame-RMS on the locked grid
        env = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop).ravel()
        x = env[None, :]
        names = ['env_rms']

    elif feature == 'rectify':
        # maddox 2016 style is keep positive and negative
        # Regressor A: keep positive peaks
        reg_pos = np.clip(y, 0, None)  # == np.maximum(y, 0)
        # Regressor B: keep inverted negative peaks
        reg_neg = np.clip(-y, 0, None)
        base_fs = sr  # in this special case base fs is different
        x = np.array([reg_pos, reg_neg])
        names = ['pos_rect', 'neg_rect']

    elif feature in ['mel', 'flux', 'mel_onsets']:
        s = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=n_mels, fmin=fmin, fmax=min(fmax, 0.5 * sr), power=1.0
        )
        if feature == 'mel':
            x = s
            centers = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
            names = [f'mel_{int(round(fc))}Hz' for fc in centers]

        elif feature == 'flux':
            odf = librosa.onset.onset_strength(
                S=librosa.power_to_db(np.maximum(s, 1e-12), 
                                      ref=np.max), sr=sr, hop_length=hop
            )
            x = odf[None, :]
            names = ['flux']
        elif feature == 'mel_onsets':
            s_db = librosa.power_to_db(np.maximum(s, 1e-12), ref=np.max)
            x = librosa.onset.onset_strength_multi(
                S=s_db,
                sr=sr,
                hop_length=hop,
                aggregate=False,
            )  # [n_mels, n_frames]
            centers = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
            names = [f'mel_{int(round(fc))}Hz' for fc in centers]

    else:
        raise ValueError("feature must be one of {'envelope','mel','flux'}")

    # 2) one LP for ALL features (time axis = 0)
    cutoff = min(max(cutoff_hz, 1e-6), 0.49 * base_fs)
    sos = butter(4, cutoff, btype='low', fs=base_fs, output='sos')
    x = sosfiltfilt(sos, x, axis=1)

    # 3) optional polyphase resampling to hit EXACT target_fs
    if not np.isclose(base_fs, target_fs):
        # r = Fraction(float(target_fs) / float(base_fs)).limit_denominator(1000)
        # X = resample_poly(X, up=r.numerator, down=r.denominator, axis=0)
        fs = float(target_fs)
        x = _resample_poly_exact(x, base_fs, target_fs)
    else:
        fs = float(base_fs)

    t = np.arange(x.shape[1]) / fs
    return x, t, names, fs


# %%
