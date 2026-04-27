# In a nutshell

[![PyPI](https://img.shields.io/pypi/v/mescal.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/mescal.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/mescal)][pypi status]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)][License]
[![Read the documentation at https://mescal.readthedocs.io/](https://img.shields.io/readthedocs/mescal/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/matthieu-str/mescal/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/matthieu-str/mescal/graph/badge.svg?token=7VUAW95C24)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/mescal/
[read the docs]: https://mescal.readthedocs.io/
[tests]: https://github.com/matthieu-str/mescal/actions?workflow=Tests
[codecov]: https://codecov.io/gh/matthieu-str/mescal
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[doi]: https://zenodo.org/doi/10.5281/zenodo.12727521

## Purpose

_mescal_ is a Python package for the creation and management of life-cycle inventory databases and generation of 
sustainability metrics derived from Life-Cycle Assessment (LCA), to be integrated Energy System Models (ESM) in order to 
make energy transition pathways sustainability assessment more holistic, transparent and reproducible.

_mescal_ is a specialized package of the [Brightway Software Framework](https://brightway.dev/), mainly relying on the [_bw2calc_](https://github.com/brightway-lca/brightway2-calc) 
and [_wurst_](https://github.com/polca/wurst) Python packages.

_mescal_ was designed for all researchers and modellers aiming to include life-cycle assessment (LCA) in their analyses, 
without necessarily being LCA experts. On the other hand, _mescal_ is also designed for LCA experts who want to
integrate projections from ESM into their LCA studies.

![workflow of the mescal methodology](docs/pics/workflow.png "workflow")

Life-Cycle Inventory (LCI) datasets are taken from ecoinvent and possibly other sources if some of the ESM technologies 
are not covered in the ecoinvent database, e.g., [_premise_](https://linkinghub.elsevier.com/retrieve/pii/S136403212200226X) additional inventories. These LCI datasets are mapped 
to the ESM technologies and resources. Systematic transformations are operated on LCI datasets, including 
regionalization, databases harmonization, double-counting removal, and life-cycle impact assessment. 
LCA indicators are then ready to be integrated in the ESM.

## How to use _mescal_?

You can follow this [example notebook](https://github.com/matthieu-str/mescal/blob/master/examples/tutorial.ipynb) to learn how to use _mescal_. It presents a real application using the 
[EnergyScope](https://www.energyscope.net/latest/) model. 

If you use _mescal_ in a scientific publication, please cite [this paper](https://link.springer.com/article/10.1007/s44498-026-00005-3):

Souttre, M., Majeau-Bettez, G., Maréchal, F., Margni, M., 2026. mescal: a tool for coupling energy system models with life-cycle assessment. J. Ind. Ecol. https://doi.org/10.1007/s44498-026-00005-3

You can also specify the version of _mescal_ you used in your publication, e.g., by including the DOI of the version you used, which can be found on [Zenodo](https://doi.org/10.5281/zenodo.12727521).

## Documentation
The documentation for _mescal_ can be found at [https://mescal.readthedocs.io/en/latest/](https://mescal.readthedocs.io/en/latest/). It mainly contains set-up instructions, examples of input data files, and example notebooks on how to use _mescal_. 

## Requirements

- **Python 3.10 or more** 
- License for [ecoinvent 3](https://ecoinvent.org/). The ecoinvent database is not included in this package. You may also check 
ecoinvent's [GDPR & EULA](https://ecoinvent.org/gdpr-eula/). 

## Installation

You can install _mescal_ via [pip] from [PyPI].
We recommend installing the optional package _pypardiso_ to speed up matrix calculations.

Install with _brightway2_-compatible libraries:
```console
$ pip install "mescal[bw2]" pypardiso
```

Install with newer _brightway25_-compatible libraries:
```console
$ pip install "mescal[bw25]" pypardiso
```

For local development (editable install) from a cloned repository using `bw2`:
```console
$ pip install -e ".[bw2]"
```
Or using `bw25`:
```console
$ pip install -e ".[bw25]"
```

If Brightway libraries are already installed and should not be touched:
```console
$ pip install -e . --no-deps
```

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide][Contributor Guide].

## License

Distributed under the terms of the [MIT license][License],
_mescal_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue][Issue Tracker] along with a detailed description.

## Support
Contact [matthieu.souttre@polymtl.ca](mailto:matthieu.souttre@polymtl.ca)

<!-- github-only -->

[command-line reference]: https://mescal.readthedocs.io/en/latest/usage.html
[License]: https://opensource.org/licenses/MIT
[Contributor Guide]: https://github.com/matthieu-str/mescal/blob/master/CONTRIBUTING.md
[Issue Tracker]: https://github.com/matthieu-str/mescal/issues