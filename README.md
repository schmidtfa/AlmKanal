## Welcome to AlmKanal

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Coverage Status](https://coveralls.io/repos/github/schmidtfa/AlmKanal/badge.svg?branch=main)](https://coveralls.io/github/schmidtfa/AlmKanal?branch=main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


AlmKanal is the M/EEG preprocessing pipeline developed at the Salzburg Brain Dynamics Lab lead by Nathan Weisz.
The Pipeline is set-up as a number of building blocks that take you automatically from your raw M/EEG recording to preprocessed Raw or Epoched data in sensor or source space.

The aim is to give you - the researcher - an easy way to quickly build your preprocessing pipeline saving you (and me) from copying annoying boilerplate code, while retaining the highest amount of flexibility to build your pipeline.
Ideally this minimzes your time spent in front of the computer preprocessing your data, while maximizing your time spent outside swimming [at the Almkanal ;)]

The only thing you need to know to get started is some MNE Python and one or two things about classes in Python.
Or you just use one of our template pipelines directly (link) :)


Currently you can install the AlmKanal package only directly from github. You just need to add the following to your pixi.toml or use pip.

[pypi-dependencies]
almkanal = { git = "https://github.com/schmidtfa/AlmKanal.git"}

