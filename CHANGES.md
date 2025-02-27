# `mescal` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - TBD

### Added
- Add a spatialized biosphere database and the regionalized EF 3.1 methods to your brightway project (in `EF 3.1 regionalized.ipynb`). Your ecoinvent database needs to be spatialized (see [`Regioinvent`](https://github.com/CIRAIG/Regioinvent/) or [`Regiopremise`](https://github.com/matthieu-str/Regiopremise/tree/master)). 
- The function `add_carbon_capture_to_plant` (in `modify_inventory.py`) adds a carbon capture and storage (CCS) process to a power plant and modifies its direct emissions accordingly.
- The `main_database_name` argument of the `ESM` class specifies the name of your ecoinvent database in case the input database in an aggregation of several databases.
- The `assessment_type` argument in `normalize_lca_metrics` (in `normalization.py`) and `generate_mod_file_ampl` (in `generate_lcia_obj_ampl.py`) functions to differentiate between life-cycle impacts computation (by default) and territorial impact computation (`assessment_type="direct emissions"`). 
- Plotting methods (`Plot` class in `plot.py`) to visualize the results of the `compute_impact_scores` function (in `impact_assessment.py`).
- The `connect_esm_results_to_database` method (in `esm_back_to_lca.py`) to connect the LCI datasets created from ESM results to another LCI database. The `create_new_database_with_esm_results` method only creates a new database with the LCI datasets from ESM results.

### Changed
- Corrected error occurring in `correct_esm_and_lca_efficiency_differences` (in `adapt_efficiency.py`) when several input fuel flows were present in an operation LCI dataset.
- The `mapping_ecoinvent_version_ipynb` notebook (to change the ecoinvent version of your mapping file) has been updated to work with version 3.10.1 of ecoinvent.
- Variables and parameters naming for direct emissions indicators in the AMPL files (in `normalization.py` and `generate_lcia_obj_ampl.py`) has been updated to be more explicit.

### Removed
- The `spatialized_database` argument of the `ESM` class was removed. It is not set to True if `spatialized_biosphere_db` is not None.

## [1.1.1] - 2024-10-11

### Added
- Arguments `specific_lcia_methods`, `specific_lcia_categories` and `specific_lcia_abbrev` in the `normalize_lca_metrics` (in `normalization.py`) and `generate_mod_file_ampl` (in `generate_lcia_obj_ampl.py`) functions to specify the LCIA methods or impact categories to be used when generating the AMPL .mod and .dat files.

### Changed
- Set the `refactor` parameter in the `normalize_lca_metrics` function (in `normalization.py`) depending on the `AoP` (in `impact_abbrev.csv`) rather than the impact category. 

## [1.1.0] - 2024-10-09

### Added
- The function `change_direct_carbon_emissions_by_factor` (in `modify_inventory.py`) changes all direct carbon emissions by a factor.
- The function `add_carbon_dioxide_flow` (in `modify_inventory.py`) adds a fossil carbon dioxide flow to an LCI dataset.

### Changed
- **Major update**: the `mescal` package has been restructured from functional programming to object-oriented programming. The classes `ESM` (in `esm.py`), `Database` (in `database.py`) and `Dataset` (in `database.py`) have been created to perform operations on the ESM database, the LCI databases and the LCI datasets respectively. All functions from v1.0.3 are still available but have been reorganized in the classes. 

## [1.0.3] - 2024-09-27

### Added
- The function `change_fossil_carbon_flows_of_biofuels` (in `modify_inventory.py`) changes fossil carbon emissions to biogenic carbon emissions if biofuels are used in the foreground.
- Metadata has been added at the beginning of AMPL files (in `normalized_lca_metric`  from `normalization.py` and `generate_lci_obj`  from `generate_lci_obj_ampl.py`). 
- Correction of CA-QC heat flows in the ecoinvent database for global markets (in `remove_quebec_flow_in_global_heat_market` from `modify_inventory.py`).
- Option added to check for duplicated activities in terms of (product, name, location) tuple when merging several databases (in `merge_databases` from `utils.py`).
- A specific category for imports and exports has been added in `technology_specifics.csv` to avoid their regionalization to the ESM geography.

### Changed
- The efficiency harmonization functions (in `adapt_efficiency.py`) can now accept multiple types of input flows instead of just one.
- The arguments of the regionalization functions have been renamed to differentiate between regionalization and spatialization.
- Renamed AMPL files (by default, `techs_lcia.dat` and `objectives.mod` are used regardless of the LCIA method employed).
- For AMPL files generation, the `refactor` parameter is computed based on the LCA indicator data rather than being a user input. 

## [1.0.2] - 2024-07-26

### Added
- Regionalization of biosphere flows in the `regionalize_activity_foreground` function (from `regionalization.py`), and added conditions to prevent the regionalization when unnecessary.
- Correction of efficiency differences between the ESM and LCI databases (scale direct emissions accordingly) in the `correct_esm_and_lca_efficiency_differences` function (from `adapt_efficiency.py`).
- Option to return the database instead of writing it (or both) in the `create_esm_database` (from `double_counting.py`) and `create_new_database_with_esm_results` (from `esm_back_to_lca.py`) functions.

### Changed 
- `concatenate_database` function renamed to `load_multiple_databases` to avoid confusion with new `merge_databases` function (from `utils.py`).
- `unit_conversion.csv` file has its columns `From` and `To` renamed to `ESM` and `LCA` for better clarity.
- Reformatted and added new information to the double-counting removal output csv files.

## [1.0.1] - 2024-07-13

### Added
- Perform double-counting removal using the `create_esm_database` function (from `double_counting.py`).
- Compute your LCA metrics for a set of impact categories using the `compute_impact_scores` function (from `impact_assessment.py`).
- Reformat results as `.mod` and `.dat` files if you are using [AMPL](https://ampl.com/) with the `normalize_lca_metrics` (from `normalization.py`) and `gen_lcia_obj` (from `generate_lci_obj_ampl.py`) functions. 
- Feed the ESM results back in the LCI database using the `create_new_database_with_esm_results` function (from `esm_back_to_lca.py`).
- Add CPC categories to additional LCI datasets, e.g., from [Premise](https://premise.readthedocs.io/en/latest/introduction.html), with the `create_new_database_with_CPC_categories` function (from `CPC.py`).
- Relink a mapping to a new database using the `create_complementary_database` function (from `link_to_premise.py`).
- Relink a mapping to a new version of [ecoinvent](https://ecoinvent.org/) using the `change_ecoinvent_version_mapping` and `update_unit_conversion_file` functions (`change_ecoinvent.py`).
- Regionalize the foreground of LCI datasets by setting `regionalize_foregrounds=True` in the `create_esm_database` function (from `double_counting.py`).
