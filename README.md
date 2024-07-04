# mescal

[![PyPI](https://img.shields.io/pypi/v/mescal.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/mescal.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/mescal)][pypi status]
[![License](https://img.shields.io/pypi/l/mescal)][license]

[![Read the documentation at https://mescal.readthedocs.io/](https://img.shields.io/readthedocs/mescal/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/matthieu-str/mescal/actions/workflows/python-test.yml/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/matthieu-str/mescal/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/mescal/
[read the docs]: https://mescal.readthedocs.io/
[tests]: https://github.com/matthieu-str/mescal/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/matthieu-str/mescal
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Requirements

- **Python 3.11**
- Licence for [ecoinvent 3](https://ecoinvent.org/). The ecoinvent database is not included in this package. You may also check ecoinvent's [GDPR & EULA](https://ecoinvent.org/gdpr-eula/). 

## Installation

You can install _mescal_ via [pip] from [PyPI]:

```console
$ pip install mescal
```

## How to use _mescal_?

You can follow this [example notebook](https://github.com/matthieu-str/mescal/blob/master/examples/tutorial.ipynb) to learn how to use _mescal_.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide][Contributor Guide].

## License

Distributed under the terms of the [Apache 2.0 license][License],
_mescal_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue][Issue Tracker] along with a detailed description.


<!-- github-only -->

[command-line reference]: https://mescal.readthedocs.io/en/latest/usage.html
[License]: https://github.com/matthieu-str/mescal/blob/main/LICENSE
[Contributor Guide]: https://github.com/matthieu-str/mescal/blob/main/CONTRIBUTING.md
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