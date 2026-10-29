---
title: AlmKanal
hide:
  - toc
---

<div class="ak-home" markdown>
<div class="grid ak-hero" markdown>
<div markdown>

Built on MNE-Python &nbsp; &middot; &nbsp; Open source
{ .ak-eyebrow }

# Alm<span>Kanal</span>

Flexible, modular and extendable preprocessing pipelines for M/EEG data
{ .ak-tagline }

Build the pipeline you need. See every step. Then turn what you ran into your Methods section.

[Get started &rarr;](quickstart.md){ .md-button .md-button--primary }
[Create your own step](custom-steps.md){ .md-button }

Your data. Your steps. No fuss.
{ .ak-note }

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

    2. Keil, A., Debener, S., Gratton, G., Junghöfer, M., Kappenman, E. S., Luck, S. J., Luu, P., Miller, G. A., & Yee, C. M. (2014). Committee report: Publication guidelines and recommendations for studies using EEG and MEG. Psychophysiology, 51(1), 1–21. https://doi.org/10.1111/psyp.12147

    > **Disclaimer**
    >
    > AlmKanal renders Methods descriptions from saved processing history and Jinja templates. Review the generated text and describe any earlier processing separately.

    </div>

    [How Methods reporting works &rarr;](methods.md)

</div>
</div>

<div class="grid ak-flow" markdown>

**01 &nbsp; Compose**<br>Reusable Python steps

**02 &nbsp; Check**<br>Declared ordering rules

**03 &nbsp; Run**<br>Process and inspect

**04 &nbsp; Document**<br>Generate Methods text

</div>

## Made for the way research evolves.

Start small, adapt a step, or bring an existing analysis along.

<div class="grid cards" markdown>

-   **Pipelines you can read**

    Steup your processing steps directly in Python, in the order they should run.

    [Build a pipeline &rarr;](pipelines.md)

-   **Steps you can extend**

    Add a custom Step without stepping away from the core framework.

    [Create your own step &rarr;](custom-steps.md)

-   **Checks before you run**

    Conflicts are catched by the ordering rules declared in the preprocessing steps.

    [Understand validation &rarr;](validation.md)

-   **Methods from your pipeline**

    Turn recorded preprocessing settings into a draft of your Methods section to review and edit.

    [Generate Methods text &rarr;](methods.md)

-   **Start with the data you have**

    Use compatible steps on existing MNE Raw or Epochs objects, including previously processed recordings.

    [Work with existing data &rarr;](pipelines.md)

-   **Bring recordings together**

    Prepare synchronized M/EEG, VPixx eye-tracking, and WAV audio data for subsequent analyses.

    [Eye tracking &rarr;](eye-tracking.md) &nbsp; [Audio alignment &rarr;](audio-alignment.md)

</div>

## One small example. The whole workflow.

Try a pipeline on synthetic EEG, save its processing history, and generate a Methods draft.

[Open the quick start &rarr;](quickstart.md){ .md-button .md-button--primary }
[Explore the examples](examples.md){ .md-button }

---

The name is a nod to Salzburg's *Almkanal* &mdash; and to the hope of spending less time rewriting preprocessing boilerplate and more time outside in the sun. ☀️ 🍹 🏖️

</div>
