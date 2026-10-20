"""Synthetic-data API example; run with AlmKanal installed.

Example settings demonstrate the interface, not a study protocol.
"""
from pathlib import Path

import mne
import numpy as np
from almkanal import AlmKanal, Filter, Resample, preprocessing_report


def make_raw() -> mne.io.BaseRaw:
    """Return a small synthetic EEG recording without downloading data."""
    rng = np.random.default_rng(42)
    sfreq = 500.0
    times = np.arange(int(30 * sfreq)) / sfreq
    signal = 2e-6 * np.sin(2 * np.pi * 10 * times)
    data = signal[None, :] + rng.normal(0, 1e-6, (4, len(times)))
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")
    return raw


def main() -> None:
    out = Path("almkanal-demo")
    out.mkdir(exist_ok=True)
    raw = make_raw()
    pipeline = AlmKanal([
        Filter(highpass=1.0, lowpass=40.0),
        Resample(sfreq=200),
    ])
    processed, report = pipeline.run(raw.copy())
    report.save(out / "quality-control.html", overwrite=True, open_browser=False)
    processed.save(out / "processed_raw.fif", overwrite=True)
    history = out / "preprocessing.json"
    pipeline.generate_json(str(history))
    methods = preprocessing_report(files=[history], out_path=out / "methods.md")
    if processed.info["sfreq"] != 200:
        raise AssertionError("Unexpected output sampling frequency.")
    if not methods.read_text(encoding="utf-8").strip():
        raise AssertionError("Methods output is empty.")
    print(methods.read_text(encoding="utf-8"))
    print(f"Outputs saved in {out.resolve()}")


if __name__ == "__main__":
    main()
