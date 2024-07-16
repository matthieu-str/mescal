# `mescal` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2024-07-13

### Added
- Perform double-counting removal using the `create_esm_database` function
- Compute your LCA metrics for a set of impact categories using the `compute_impact_scores` function
- Reformat the results as `.mod` and `.dat` files if you are using [AMPL](https://ampl.com/) using the `normalize_lca_metrics` and `gen_lcia_obj` functions 
- Feed the ESM results back in the LCI database using the `create_new_database_with_esm_results` function
- Add CPC categories to additional LCI datasets, e.g., from [Premise](https://premise.readthedocs.io/en/latest/introduction.html), using the `create_new_database_with_CPC_categories` function
- Relink a mapping to a new database / a new version of [ecoinvent](https://ecoinvent.org/) using the `create_complementary_database` function
- Regionalize the foreground of LCI datasets by setting `regionalize_foregrounds=True` in the `create_esm_database` function
