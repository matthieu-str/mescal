# `mescal`

[![PyPI](https://img.shields.io/pypi/v/mescal.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/mescal.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/mescal)][pypi status]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)][License]
[![Read the documentation at https://mescal.readthedocs.io/](https://img.shields.io/readthedocs/mescal/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/matthieu-str/mescal/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/matthieu-str/mescal/graph/badge.svg?token=7VUAW95C24)][codecov]
[![DOI](https://zenodo.org/badge/813273884.svg)][doi]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/mescal/
[read the docs]: https://mescal.readthedocs.io/
[tests]: https://github.com/matthieu-str/mescal/actions?workflow=Tests
[codecov]: https://codecov.io/gh/matthieu-str/mescal
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[doi]: https://zenodo.org/doi/10.5281/zenodo.12727521

## What is `mescal`?

`mescal` is a Python package for the creation and management of life-cycle inventory databases and generation of 
sustainability metrics derived from Life-Cycle Assessment (LCA), to be integrated Energy System Models (ESM) in order to 
make energy transition pathways sustainability assessment more holistic, transparent and reproducible.

`mescal` is a specialized package of the [Brightway Software Framework](https://brightway.dev/), mainly relying on the [`bw2calc`](https://github.com/brightway-lca/brightway2-calc) 
and [`wurst`](https://github.com/polca/wurst) Python packages.

`mescal` was designed for all researchers and modellers aiming to include life-cycle assessment (LCA) in their analyses, 
without necessarily being LCA experts. On the other hand, `mescal` is also designed for LCA experts who want to
integrate projections from ESM into their LCA studies.

![workflow of the mescal methodology](docs/pics/workflow_v2.png "workflow")

Life-Cycle Inventory (LCI) datasets are taken from ecoinvent and possibly other sources if some of the ESM technologies 
are not covered in the ecoinvent database, e.g., [`premise`](https://linkinghub.elsevier.com/retrieve/pii/S136403212200226X) additional inventories. These LCI datasets are mapped 
to the ESM technologies and resources. Systematic transformations are operated on LCI datasets, including 
regionalization, databases harmonization, double-counting removal, and life-cycle impact assessment. 
LCA indicators are then ready to be integrated in the ESM.

## How to use `mescal`?

You can follow this [example notebook](https://github.com/matthieu-str/mescal/blob/master/examples/tutorial.ipynb) to learn how to use `mescal`. It presents a real application using the 
[EnergyScope](https://library.energyscope.ch/main/) model. 

## Requirements

- **Python 3.10 or more** 
- Licence for [ecoinvent 3](https://ecoinvent.org/). The ecoinvent database is not included in this package. You may also check 
ecoinvent's [GDPR & EULA](https://ecoinvent.org/gdpr-eula/). 

## Installation

You can install `mescal` via [pip] from [PyPI]:

```console
$ pip install mescal
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide][Contributor Guide].

## License

Distributed under the terms of the [MIT license][License],
`mescal` is free and open source software.

## Issues

If you encounter any problems,
please [file an issue][Issue Tracker] along with a detailed description.


<!-- github-only -->

[command-line reference]: https://mescal.readthedocs.io/en/latest/usage.html
[License]: https://opensource.org/licenses/MIT
[Contributor Guide]: https://github.com/matthieu-str/mescal/blob/master/CONTRIBUTING.md
[Issue Tracker]: https://github.com/matthieu-str/mescal/issues


## Building the Documentation

You can build the documentation locally by installing the documentation Conda environment:

```bash
conda env create -f docs/environment.yaml
```

activating the environment

```bash
conda activate sphinx_mescal
```

and [running the build command](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#sphinx-build):

```bash
sphinx-build docs _build/html --builder=html --jobs=auto --write-all; open _build/html/index.html
```