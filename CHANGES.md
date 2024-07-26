# `mescal` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2024-07-26

### Added
- regionalization of biosphere flows in the `regionalize_activity_foreground` function, and added conditions to prevent the regionalization when unnecessary
- correction of efficiency differences between the ESM and LCI databases (scale direct emissions accordingly) in the `correct_esm_and_lca_efficiency_differences` function
- possibility to return the database instead of writing it (or both) in the `create_esm_database` and `create_new_database_with_esm_results` functions

### Changed 
- `concatenate_database` function renamed `load_multiple_databases` to avoid confusion with new `merge_databases` function
- `unit_conversion.csv` file has its columns `From` and `To` renamed `ESM` and `LCA` for better clarity
- Reformatted and added new information to the double-counting removal output csv files

## [1.0.1] - 2024-07-13

### Added
- Perform double-counting removal using the `create_esm_database` function
- Compute your LCA metrics for a set of impact categories using the `compute_impact_scores` function
- Reformat the results as `.mod` and `.dat` files if you are using [AMPL](https://ampl.com/) using the `normalize_lca_metrics` and `gen_lcia_obj` functions 
- Feed the ESM results back in the LCI database using the `create_new_database_with_esm_results` function
- Add CPC categories to additional LCI datasets, e.g., from [Premise](https://premise.readthedocs.io/en/latest/introduction.html), using the `create_new_database_with_CPC_categories` function
- Relink a mapping to a new database using the `create_complementary_database` function
- Relink a mapping to a new version of [ecoinvent](https://ecoinvent.org/) using the `change_ecoinvent_version_mapping` and `update_unit_conversion_file` functions
- Regionalize the foreground of LCI datasets by setting `regionalize_foregrounds=True` in the `create_esm_database` function
