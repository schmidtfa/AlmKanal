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
Before Magnetoencephalography (MEG) and electroencephalography (EEG) can be used to investigate neural processes, they generally undergo several processing steps, such as e.g. filtering, artifact correction, epoching, and source reconstruction. The choice and order of these steps depends on the recording setup and the research question. Consequently, M/EEG preprocessing often requires substantial custom code, which can be difficult to maintain, reuse, and describe consistently in scientific publications.
AlmKanal is an open-source Python package for building flexible, modular, and extendable M/EEG preprocessing pipelines. Built around MNE-Python data structures [@gramfort2014mne], it allows researchers to compose processing steps directly in Python and to add custom procedures without changing the core framework. Pipelines can be applied to continuous (Raw) or epoched (Epochs) data, including recordings that have already been partly preprocessed. AlmKanal checks supported constraints on processing order and can automatically generate human-readable descriptions of executed steps for scientific Methods sections.

# Statement of Need
M/EEG preprocessing commonly consists of a sequence of (partly) interdependent operations. Although general-purpose analysis libraries such as MNE-Python provide most of the necessary algorithms [@gramfort2014mne], the choice, order, and parameters of preprocessing steps depend on the characteristics of the data and the scientific question. Consequently, preprocessing workflows need to be flexible enough to accommodate study-specific requirements, while ensuring that individual processing steps are applied consistently and reproducibly. 

To accommodate these requirements, researchers frequently develop study-specific preprocessing scripts. However, these can involve substantial boilerplate code, and reusing processing steps across workflows may introduce errors or inconsistencies. Furthermore, keeping scientific documentation aligned with the actual preprocessing performed can be challenging. Thus, there is a need for tools that combine flexible pipeline construction with reusable, reliable, and well-documented processing steps.

AlmKanal addresses these challenges through scikit-learn-inspired [@sklearn], pipelines composed of reusable and easily extendable processing steps. Pipelines can be applied to pre-loaded MNE-Python Raw or Epochs objects without requiring extensive configuration files or a fixed starting point. AlmKanal also internally validates the logical order of the passed processing steps and automatically generates Methods descriptions, facilitating consistent preprocessing and scientific documentation.

# State of the field
Several tools support M/EEG preprocessing. MNE-Python provides the underlying algorithms [@gramfort2014mne], while MNE-BIDS-Pipeline offers automated, configuration-driven workflows [@mnebidspipe]. Other frameworks, including osl-ephys [@vanEs] and Eelbrain [@eelbrain], also support customizable preprocessing pipelines.

AlmKanal complements these tools by emphasizing flexible, Python-native pipelines with easily extendable processing steps. Its distinguishing features include validation of processing order and automatic generation of preprocessing Methods text, rather than only quality-control reports or dataset descriptions. 

# Software design
Inspired by scikit-learn's pipeline architecture [@sklearn], AlmKanal combines reusable processing steps into transparent, Python-native workflows built on MNE-Python. Users can easily add custom steps without modifying the framework or relying on extensive configuration files.

AlmKanal validates the logical order of processing steps and automatically generates Methods descriptions to facilitate reproducible scientific reporting.

# AI usage disclosure
OpenAI's ChatGPT was used to assist with code review, the development of the logic underlying the Jinja-based automatic Methods reporting functionality, and the drafting and editing of this manuscript as well as the project's documentation website. All AI-assisted contributions were reviewed and verified by the authors.

# Acknowledgements
This project received financial support from the Land Salzburg through the BrainAge project (20102/F2400537-FPR).

# References