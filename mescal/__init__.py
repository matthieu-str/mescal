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
)

__version__ = "1.0.3"

from .database import Database, Dataset
from .esm import ESM
from .utils import random_code, ecoinvent_unit_convention, change_mapping_year
