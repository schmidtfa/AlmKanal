---
title: AlmKanal
hide:
  - toc
---

<div class="ak-home" markdown>
<div class="grid ak-hero" markdown>
<div markdown>

# Alm<span>Kanal</span>

Flexible, modular and extendable preprocessing pipelines for M/EEG data
{ .ak-tagline }

Build and run preprocessing pipelines, keep track of the settings, and generate a Methods draft from your processing history.

[Get started &rarr;](quickstart.md){ .md-button .md-button--primary }
[Create your own step](custom-steps.md){ .md-button }

</div>
<div class="ak-demo" markdown>

From code to Methods
{ .ak-eyebrow }

=== "Python pipeline"

    ```python
    from almkanal import (
        AlmKanal, Filter, Resample,
        preprocessing_report,
    )

    pipeline = AlmKanal(steps=[
        Filter(highpass=1.0, lowpass=40.0),
        Resample(sfreq=200),
    ])

    # raw: your existing MNE Raw object
    cleaned, qc = pipeline.run(raw.copy())
    pipeline.generate_json("history.json")
    preprocessing_report(
        files=["history.json"],
        out_path="methods.md",
    )
    ```

    Assumes an existing `raw` object. The [quick start](quickstart.md) includes a complete example with synthetic data.
    { .ak-note }

=== "Methods draft"

    <div class="ak-methods-preview" markdown>

    ### Preprocessing

    #### Offline filtering

    A band-pass filter (1–40 Hz) was applied to the continuous data. The finite impulse response (FIR) filter was designed using firwin with a hamming window. A symmetric linear-phase filter with delay compensation was used, resulting in zero-phase, non-causal filtering. The realized filter length was 661 samples (3.305 s), corresponding to FIR order 660. The lower and upper transition bandwidths were 1.0 Hz and 10.0 Hz, respectively. The −6 dB cutoff frequencies were 0.5 Hz and 45.0 Hz, located at the midpoints of the lower and upper transition bands. Filter design and reporting was aimed to follow recommended practices for electrophysiological data (1, 2).

    #### Resampling

    The continuous data were resampled from 1000.0 Hz to 200 Hz using FFT-based resampling with a boxcar frequency-domain window.

    ---

    #### References

    1. Widmann, A., Schröger, E., & Maess, B. (2015). Digital filter design for electrophysiological data: A practical approach. Journal of Neuroscience Methods, 250, 34–46. doi:10.1016/j.jneumeth.2014.08.002

    2. Keil, A., Debener, S., Gratton, G., Junghöfer, M., Kappenman, E. S., Luck, S. J., Luu, P., Miller, G. A., & Yee, C. M. (2014). Committee report: Publication guidelines and recommendations for studies using EEG and MEG. Psychophysiology, 51(1), 1–21. [https://doi.org/10.1111/psyp.12147](https://doi.org/10.1111/psyp.12147)

    > **Disclaimer**
    >
    > AlmKanal renders Methods descriptions from saved processing history and Jinja templates. Review the generated text and describe any earlier processing separately.

    </div>

</div>
</div>

## What you can do with AlmKanal

AlmKanal builds on MNE-Python by bringing commonly used preprocessing functions together in a modular pipeline framework. Feel free to use the existing steps, add your own, or start with data that have already been processed.

<div class="grid cards" markdown>

-   **Pipelines in Python**

    Define and configure processing steps directly in Python. The pipeline makes the order of operations explicit.

    [Build a pipeline &rarr;](pipelines.md)

-   **Your own processing steps**

    Add a custom `AlmKanalStep` when you need an operation that is not covered by the built-in steps.

    [Create your own step &rarr;](custom-steps.md)

-   **Step-order checks**

    Steps can declare which operations must come before or after them. AlmKanal checks those constraints before processing starts.

    [Understand validation &rarr;](validation.md)

-   **A record of what ran**

    Save the preprocessing settings used in a pipeline and generate an editable draft of the Methods section from that history.

    [Generate Methods text &rarr;](methods.md)

-   **Use existing MNE data**

    Apply compatible steps to MNE `Raw` or `Epochs` objects, including recordings that have already been preprocessed.

    [Work with existing data &rarr;](pipelines.md)

-   **Audio and eye-tracking alignment**

    Tools for working with M/EEG recordings alongside WAV audio stimuli and VPixx eye-tracking data.

    [Eye tracking &rarr;](eye-tracking.md) &nbsp; [Audio alignment &rarr;](audio-alignment.md)

</div>

## Try a complete example

The [quick start](quickstart.md) runs a pipeline on synthetic EEG data, saves the processing history, and generates a Methods draft.

[Open the quick start &rarr;](quickstart.md){ .md-button .md-button--primary }
[Explore the examples](examples.md){ .md-button }

---

The name is a nod to Salzburg's *Almkanal* — and to the hope of spending less time rewriting preprocessing code and more time outside in the sun. ☀️ 🍹 🏖️

</div>
