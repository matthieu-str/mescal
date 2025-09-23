# `mescal` Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-09-23

### Added
- The `background_double_counting_removal` method (in `double_counting.py`) that creates a new version of the technosphere matrix where flows included in the ESM end-use demands are set to zero, thus preventing double-counting. This step is needed if the ESM end-use demands include the production and operation of new infrastructures. 
- The `req_technosphere` argument to the `compute_impact_scores` method (in `impact_assessment.py`). If True, the method returns the technosphere flows requirements (life-cycle inventory) of the ESM technologies and resources. 
- The `esm_end_use_demands` and `remove_double_counting_to` arguments to the `ESM` class in order to apply the double-counting removal process to infrastructure and resource datasets (in addition to operation datasets).
- The `stop_background_search_when_first_flow_found` argument in the `ESM` class to stop or continue the background search for double-counting removal when the first flow of the targeted category is found (in `esm.py`).
- The `_validation_double_counting` method (in `double_counting.py`, included in the `create_esm_database` method) to compare the flows removed during the double-counting removal process and the flows in the ESM.
- The `validation_direct_carbon_emissions` method (in `impact_assessment.py`) comparing direct carbon emissions in the ESM and from the impact assessment phase (direct emissions module).

### Changed
- The type of the `regionalize_foregrounds` argument in the `ESM` class is now a string or list of string (instead of a boolean) in order to apply the foreground regionalization process only to certain types of LCI datasets (instead of all or none of them). 
- The `harmonize_with_esm` argument of the `create_new_database_with_esm_results` method has been replaced by the two arguments `harmonize_efficiency_with_esm` and `harmonize_capacity_factor_with_esm` to be able to harmonize the efficiency and capacity factor separately (in `esm_back_to_lca.py`).

## [1.2.0] - 2025-07-08

### Added
- `PathwayESM` class (in `esm.py`) to create databases and impact score dataframes for all ESM time steps at once.
- Contribution analysis of processes in `compute_impact_scores` (in `impact_assessment.py`) by setting `contribution_analysis` to `processes`.
- `max_depth_double_counting_search` argument in the `ESM` class to limit the depth of the search in the double-counting removal process (in `double_counting.py`).
- Harmonization of efficiency and capacity factor with the ESM in the ESM results database (in `esm_back_to_lca.py` and `adapt_efficiency.py`).
- Loss coefficient (based on the one of the original dataset) for datasets in the ESM results database (in `esm_back_to_lca.py`).

### Changed
- Corrected unit conversion for elementary flows contribution analysis in `compute_impact_scores` (in `impact_assessment.py`).
- Improved double-counting removal process for market-type activities in the ESM results database (i.e., removal of infrastructure flows) (in `esm_back_to_lca.py` and `double_counting.py`).
- If specified in `tech_specifics`, double-counting removal during background search is allowed even if a flows has already been set to zero (in `double_counting.py`).
- Relinking of the main database with the ESM results database can be based on activity names and/or product names, and for a user-defined set of geographies `locations` (in `esm_back_to_lca.py`).
- Default `esm_location` is GLO and default `accepted_locations` is [`esm_location`] in the `ESM` class (in `esm.py`).

## [1.1.4] - 2025-04-09

### Added
- Contribution analysis of elementary flows in the `MultiLCA` class of `compute_impact_scores` (in `impact_assessment.py`) by setting `contribution_analysis` to True.
- Use of [`logging`](https://docs.python.org/3/library/logging.html) and [`tqdm`](https://tqdm.github.io/) for improved logging and progress tracking in the `create_esm_database` function (in `double_counting.py`), `compute_impact_scores` function (in `impact_assessment.py`), and `create_new_database_with_esm_results`, `connect_esm_results_to_database` functions (in `esm_back_to_lca.py`).
- Check for duplicates in input dataframes in the `check_inputs` function (in `esm.py`).

### Changed
- Corrected tuple item assignment for creating the direct emissions database in `compute_impact_scores` (in `impact_assessment.py`).

## [1.1.3] - 2025-03-29

### Added
- A mapping between existing _ecoinvent_ products and their CPC categories (`data/mapping_existing_products_to_CPC.json`) and a method to add CPC categories, when missing, based on this mapping (`add_CPC_categories_based_on_existing_activities` in `CPC.py`).
- Handling error due to missing category ('air', 'lower stratosphere + upper troposphere') for "Carbon dioxide, non-fossil" in _ecoinvent_ 3.10 and earlier.

## [1.1.2] - 2025-03-11

### Added
- A spatialized biosphere database and the regionalized EF 3.1 methods (in `dev/ef regionalized/EF 3.1 regionalized.ipynb`). Your ecoinvent database then needs to be spatialized (see [`Regioinvent`](https://github.com/CIRAIG/Regioinvent/) or [`Regiopremise`](https://github.com/matthieu-str/Regiopremise/tree/master)). 
- The function `add_carbon_capture_to_plant` (in `modify_inventory.py`) that adds a carbon capture and storage (CCS) process to a power plant and modifies its direct emissions accordingly.
- The `main_database_name` argument of the `ESM` class that specifies the name of your ecoinvent database in case the input database in an aggregation of several databases.
- The `assessment_type` argument in `normalize_lca_metrics` (in `normalization.py`) and `generate_mod_file_ampl` (in `generate_lcia_obj_ampl.py`) functions to differentiate between life-cycle impacts computation (by default) and direct emissions impact computation (`assessment_type="direct emissions"`). 
- Plotting methods (`Plot` class in `plot.py`) to visualize the results of the `compute_impact_scores` function (in `impact_assessment.py`).
- The `connect_esm_results_to_database` method (in `esm_back_to_lca.py`) to connect the LCI datasets created from ESM results to another LCI database. The `create_new_database_with_esm_results` method now only creates a new database with the LCI datasets from ESM results.
- The `biosphere_db_name` argument in the `ESM` class to specify the name of the biosphere database.

### Changed
- Corrected error occurring in `correct_esm_and_lca_efficiency_differences` (in `adapt_efficiency.py`) when several input fuel flows were present in an operation LCI dataset.
- The `mapping_ecoinvent_version_ipynb` notebook (to change the ecoinvent version of your mapping file) has been updated to work with version 3.10.1 of ecoinvent.
- Variables and parameters naming for direct emissions indicators in the AMPL files (in `normalization.py` and `generate_lcia_obj_ampl.py`) to be more explicit.
- Inequality for equality constraints in the AMPL files (in `generate_lcia_obj_ampl.py`) in case there is no LCA objective function or constraint.  

### Removed
- The `spatialized_database` argument of the `ESM` class. It is now set to True if `spatialized_biosphere_db` is not `None`.

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
- Add CPC categories to additional LCI datasets, e.g., from [`premise`](https://premise.readthedocs.io/en/latest/introduction.html), with the `create_new_database_with_CPC_categories` function (from `CPC.py`).
- Relink a mapping to a new database using the `create_complementary_database` function (from `link_to_premise.py`).
- Relink a mapping to a new version of [ecoinvent](https://ecoinvent.org/) using the `change_ecoinvent_version_mapping` and `update_unit_conversion_file` functions (`change_ecoinvent.py`).
- Regionalize the foreground of LCI datasets by setting `regionalize_foregrounds=True` in the `create_esm_database` function (from `double_counting.py`).
