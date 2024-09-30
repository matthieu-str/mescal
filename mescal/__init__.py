"""mescal."""

__all__ = (
    "__version__",
    # Add functions and variables you want exposed in `mescal.` namespace here

    # esm_database.py
    "ESMDatabase",

    # database.py
    "Database",
    "Dataset",

    # utils.py
    "random_code",
    "ecoinvent_unit_convention",
    "change_mapping_year",

    # "change_location_mapping_file",
    # "create_complementary_database",
    # "create_new_database_with_CPC_categories",
    # "create_esm_database",
    # "compute_impact_scores",
    # "normalize_lca_metrics",
    # "gen_lcia_obj",
    # "create_or_modify_activity_from_esm_results",
    # "create_new_database_with_esm_results",
    # "change_ecoinvent_version_mapping",
    # "load_concatenated_ecoinvent_change_report",
    # "update_unit_conversion_file",
    # "correct_esm_and_lca_efficiency_differences",
    # "remove_quebec_flow_in_global_heat_market",
)

__version__ = "1.0.3"

from .database import Database, Dataset
from esm_database import ESMDatabase
from .utils import random_code, ecoinvent_unit_convention, change_mapping_year
# from .location_selection import *
# from .link_to_premise import *
# from .double_counting import *
# from .impact_assessment import *
# from .normalization import *
# from .generate_lcia_obj_ampl import *
# from .esm_back_to_lca import *
# from .change_ecoinvent import *
# from .adapt_efficiency import *
# from .modify_inventory import *
