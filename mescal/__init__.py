"""mescal."""

__all__ = (
    "__version__",
    # Add functions and variables you want exposed in `mescal.` namespace here

    # esm.py
    "ESM",

    # database.py
    "Database",
    "Dataset",

    # utils.py
    "random_code",
    "ecoinvent_unit_convention",
    "change_mapping_year",

    # change_ecoinvent.py
    "change_ecoinvent_version_mapping",
    "update_unit_conversion_file",
)

__version__ = "1.0.3"

from .database import Database, Dataset
from .esm import ESM
from .utils import *
from .change_ecoinvent import *
