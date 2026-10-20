---
title: 'AlmKanal - Flexible, modular and extendable preprocessing pipelines for M/EEG data'
tags:
    - Python
    - preprocessing
    - electrophysiology
    - electroencephalography
    - EEG
    - magnetoencephalography
    - MEG
authors:
    - name: Fabian Schmidt
      corresponding: true
      orcid: 0000-0002-9839-1614
      affiliation: 1
    - name: Gianpaolo Demarchi
      orcid: 0000-0002-7597-9298
      affiliation: 1
    - name: Alexander Weigl
      orcid: 0009-0006-8435-5846
      affiliation: 1
    - name: Annika Etzler
      orcid: 0009-0005-7926-3472x
      affiliation: 1
    - name: Larissa Reitinger
      orcid: 0000-0002-6650-9948
      affiliation: 1
    - name: Veronika Holler
      orcid: 0009-0004-9588-188X
      affiliation: 1
    - name: Nathan Weisz
      orcid: 0000-0001-7816-0037
      affiliation: "1, 2"
    - name: Thomas Hartmann
      orcid: 0000-0002-8298-8125
      affiliation: 1
affiliations:
    - name: Paris-Lodron-University of Salzburg, Department of Psychology, Centre for Cognitive Neuroscience, Salzburg, Austria
      index: 1
    - name: Neuroscience Institute, Christian Doppler University Hospital, Paracelsus Medical University, Salzburg, Austria
      index: 2
date: 20 October 2026
bibliography: paper.bib
---

# Summary
Magnetoencephalography (MEG) and electroencephalography (EEG) measure brain activity with high temporal resolution. Before these recordings can be used to investigate neural processes, they generally undergo several processing steps, such as filtering, artifact correction, epoching, and, when appropriate, source reconstruction. The choice and order of these steps depend on the recording setup and the research question. Consequently, M/EEG preprocessing often requires substantial custom code, which can be difficult to maintain, reuse, and describe consistently in scientific publications.

AlmKanal is an open-source Python package for building flexible, modular, and extendable M/EEG preprocessing pipelines. Built around MNE-Python data structures [@gramfort2014mne], it allows researchers to compose processing steps directly in Python and to add custom procedures without changing the core framework. Pipelines can be applied to continuous (Raw) or epoched (Epochs) data, including recordings that have already been partly preprocessed. AlmKanal checks supported constraints on processing order and can automatically generate human-readable descriptions of executed steps for scientific Methods sections. In addition to conventional M/EEG preprocessing, it supports temporal alignment with VPixx eye-tracking recordings and WAV audio files, facilitating subsequent multimodal, temporal response function (TRF), and coherence analyses.


# Statement of Need
M/EEG preprocessing commonly consists of a sequence of interdependent operations. Although general-purpose analysis libraries such as MNE-Python provide the necessary algorithms [@gramfort2014mne], researchers frequently assemble these operations in study-specific scripts. Such scripts can accumulate repeated code for loading data, executing processing functions, passing intermediate results between steps, and documenting the choices made. Reusing only part of an existing workflow or introducing a new processing method may therefore require extensive modifications.

AlmKanal is intended for electrophysiology researchers who wish to construct and adapt preprocessing workflows, as well as developers who implement and share new processing steps. Its architecture is inspired by the pipeline composition approach of scikit-learn [@sklearn]. Rather than separating pipeline definitions from the surrounding Python analysis into extensive external configuration files, users compose an explicit sequence of reusable processing objects. This makes the order of operations visible in the analysis code and provides a direct route for incorporating custom procedures. Text-based configurations remain useful for standardized and readily shareable workflows; AlmKanal deliberately prioritizes Python-level extensibility instead.

Importantly, a pipeline need not begin with an untouched recording. Users can apply appropriate processing steps to existing MNE-Python Raw or Epochs objects, integrating AlmKanal with earlier processing performed elsewhere. To help prevent incompatible combinations, the package validates supported logical constraints on step ordering, for example preventing sensor-space operations from being scheduled after source reconstruction where such an ordering is invalid. These checks are safeguards against known incompatibilities, not substitutes for scientific judgment about the suitability of a preprocessing strategy.

A second motivation is scientific reporting. Preprocessing sections are often written separately from the code that produced the data, creating opportunities for omissions and discrepancies. AlmKanal can generate draft Methods text from the processing steps represented in its pipelines. This facilitates documentation of the pipeline while leaving researchers responsible for checking the generated prose and describing any processing performed outside AlmKanal.

# State of the field
Several mature tools support M/EEG preprocessing. MNE-Python provides a comprehensive collection of methods for signal processing, artifact correction, and source estimation [@gramfort2014mne], but assembling and documenting a complete custom workflow remains the user's responsibility. MNE-BIDS-Pipeline automates large parts of preprocessing and analysis for BIDS-organized datasets, using configuration files and providing detailed processing reports [@mnebidspipeline; @jas2018reproducible]. This approach is particularly useful for standardized, reproducible, and large-scale processing, whereas AlmKanal targets users who want to assemble an explicit sequence of reusable Python processing steps or run selected steps on existing MNE objects.

The osl-ephys toolbox also builds on MNE-Python and supports modular preprocessing, user-defined functions, batch execution, and quality-control reports [@vanEs]. Its preprocessing chains are generally described through a configuration interface, which can be represented as YAML text or Python dictionaries. Eelbrain provides an additional Python framework for M/EEG analysis, including configurable preprocessing, source estimation, and TRF analysis [@brodbeck2023eelbrain]. Thus, neither modularity nor support for custom processing functions is unique to AlmKanal.

Instead, AlmKanal is designed around the combination of directly composed Python pipelines, straightforward extension through custom steps, checks for incompatible processing order, and automatic generation of preprocessing Methods descriptions. Reporting is a particularly relevant distinction: HTML quality-control reports and saved configurations help document an analysis, but are not the same as generating a Methods narrative describing the processing operations. For context, MNE-BIDS also offers a function for producing a Methods paragraph from dataset metadata [@appelhoff2019mnebids]; AlmKanal focuses on describing processing represented by its executable pipeline. Its VPixx eye-tracking and audio-alignment functionality additionally supports experimental workflows in which electrophysiological recordings must be synchronized with other data streams. The aim is therefore not to replace these mature toolboxes, but to provide a complementary interface for constructing, validating, and documenting customizable M/EEG workflows.

# Software design
AlmKanal represents processing stages as reusable components assembled into an ordered, Python-native pipeline. This design follows the broad principles of scikit-learn's pipeline API [@sklearn] while working with the MNE-Python data representations commonly used in electrophysiology. Each processing component can encapsulate an operation and its parameters, allowing researchers to inspect a workflow in the same order in which its steps are intended to run. New procedures can be introduced as custom components, rather than requiring modifications to a fixed end-to-end pipeline.

The choice to use Python objects instead of relying on large declarative configuration files favors interactive development, access to existing Python libraries, and integration with study-specific code. The trade-off is that text-based configurations can be easier to compare, serialize, and execute independently of Python programming knowledge. AlmKanal does not attempt to replace the low-level algorithms in MNE-Python; it supplies an orchestration and documentation layer around them.

The pipeline additionally evaluates supported ordering constraints to detect incompatible operations before or during execution. Its Methods-generation facility connects processing components with human-readable descriptions, so the processing definition and its scientific documentation are maintained closer together. Neither mechanism guarantees methodological correctness: validation covers implemented rules, and generated text must be reviewed, particularly when input data were processed beforehand. Dedicated alignment functionality for VPixx eye-tracking data and WAV audio expands the range of multimodal analyses for which these pipelines can prepare data.

# AI usage disclosure
OpenAI's ChatGPT was used to assist with code review, the development of the logic underlying the Jinja-based automatic Methods reporting functionality, and the drafting and editing of this manuscript as well as the project's documentation website. All AI-assisted code and implementation logic have been reviewed and verified for correctness by the developers. All AI-assisted contributions were reviewed and verified by the authors.

# Acknowledgements
This project received financial support from the Land Salzburg through the BrainAge project (20102/F2400537-FPR).

# References