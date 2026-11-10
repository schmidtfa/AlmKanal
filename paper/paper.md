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
Magnetoencephalography (MEG) and electroencephalography (EEG) recordings usually require several preprocessing steps before they can be analyzed. These typically aim to reduce external noise and artifacts and to prepare the data for subsequent analyses. Common procedures include filtering, artifact correction, and epoching, but may also involve more specific steps such as source reconstruction. Which steps are necessary, how they are implemented, and in which order they are applied depends on the recording setup, the characteristics of the data, and the research question. Consequently, preprocessing workflows often need to be adapted between studies.
AlmKanal is an open-source Python package for building flexible, modular and easily extendable M/EEG preprocessing pipelines. Built on MNE-Python [@gramfort2014mne], it allows users to combine processing steps directly in Python and easily add their own procedures. Furthermore, AlmKanal checks for incompatible preprocessing sequences and can automatically generate human-readable descriptions of the performed preprocessing steps for the Methods section of a paper.

# Statement of Need
M/EEG preprocessing requires a balance between flexibility and consistency. Different studies often require different preprocessing workflows, while many individual processing steps remain the same internally. Although MNE-Python provides most of the necessary algorithms [@gramfort2014mne], researchers frequently combine them in study-specific scripts. This can lead to substantial amounts of repetitive code and make it difficult to reuse individual processing steps across studies. Furthermore, modifying existing scripts can introduce errors or make it difficult to keep scientific documentation consistent with the actual preprocessing performed.
AlmKanal aims to simplify this process. Inspired by scikit-learn's pipeline architecture [@sklearn], it allows researchers to combine reusable preprocessing steps and easily implement new ones. Rather than requiring a fixed workflow or extensive configuration files, AlmKanal can be applied to already loaded `mne.io.Raw` or `mne.Epochs` objects. In addition, checks for incompatible processing sequences and automatically generated Methods descriptions help to keep preprocessing workflows and their documentation consistent.

# State of the field
Several open-source software packages exist for M/EEG preprocessing. In Python, MNE-Python [@gramfort2014mne] is among the most widely used packages [@cesnaite]. Several other packages build on MNE-Python to provide automated, customizable preprocessing pipelines, including MNE-BIDS-Pipeline [@mnebidspipe] and osl-ephys [@vanEs].
AlmKanal complements these tools by providing flexible, modular pipelines that can be defined directly in Python. The aim is to make preprocessing steps easier to reuse across different studies without requiring users to follow a fixed workflow. Users can combine individual steps, easily add their own procedures, and apply pipelines to data that have already undergone some preprocessing. Furthermore, AlmKanal checks for incompatible processing sequences and can automatically generate descriptions of the steps performed for the Methods section of a paper. Thus, the main goal of AlmKanal is not to introduce new preprocessing algorithms, but to make existing and custom procedures easier to combine, reuse, and document.

# Software design
AlmKanal's pipeline architecture is inspired by scikit-learn [@sklearn]. Individual preprocessing steps are defined as Python objects (`AlmKanalSteps`) that can be combined into a pipeline. This makes it easy to see which steps are performed and in which order. Users can also implement their own preprocessing steps without modifying the underlying framework. Defining pipelines directly in Python avoids the need for extensive configuration files and makes it easy to integrate custom procedures.
AlmKanal works with MNE-Python `mne.io.Raw` and `mne.Epochs` objects, allowing users to start a pipeline at different stages of preprocessing. To avoid incompatible processing sequences, AlmKanal checks whether the proposed order of operations is valid, for example preventing MaxFilter from being applied after source reconstruction. Furthermore, Jinja-based templates are used to automatically generate human-readable descriptions of the preprocessing steps for the Methods section of a paper. This helps to keep the documentation consistent with the actual processing performed.

# AI usage disclosure
OpenAI's ChatGPT was used to assist with code review, the development of the logic underlying the Jinja-based automatic Methods reporting functionality, and the drafting and editing of this manuscript as well as the project's documentation website. All AI-assisted contributions were reviewed and verified by the authors.

# Acknowledgements
This project received financial support from the Land Salzburg through the BrainAge project (20102/F2400537-FPR).

# References