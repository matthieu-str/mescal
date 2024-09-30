import pandas as pd
from mescal.database import Database
from mescal.esm import ESM
import pytest

dummy_db = [
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "DE",
        "unit": "kilowatt hour",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "876473"
    },
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "FR",
        "unit": "kilowatt hour",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "918932"
    },
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "RoW",
        "unit": "kilowatt hour",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "782784"
    }
]

dummy_reg_biosphere_db = [
    {
        "name": "Water, CH",
        "categories": ('water',),
        "unit": "cubic meter",
        "database": "biosphere3_regionalized_flows",
        "code": "876473",
    },
    {
        "name": "Water, RER",
        "categories": ('water',),
        "unit": "cubic meter",
        "database": "biosphere3_regionalized_flows",
        "code": "8375893",
    },
    {
        "name": "Water, GLO",
        "categories": ('water',),
        "unit": "cubic meter",
        "database": "biosphere3_regionalized_flows",
        "code": "2422322",
    },
]


@pytest.mark.tags("workflow")
def test_change_location_activity():
    esm = ESM(
        mapping=pd.DataFrame(),
        model=pd.DataFrame(),
        unit_conversion=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
        esm_db_name="esm_db",
        locations_ranking=["FR", "RER", "RoW"],
        main_database=Database(db_as_list=dummy_db),
        esm_location="FR",
    )

    updated_location = esm.change_location_activity(
        activity="market for electricity, low voltage",
        product="electricity, low voltage",
        location="DE",
        database="ecoinvent-3.9.1-cutoff",
        technosphere_or_biosphere_db=esm.main_database,
        activity_type="technosphere",
    )

    assert updated_location == "FR"


@pytest.mark.tags("workflow")
def test_change_location_activity_biosphere():
    esm = ESM(
        mapping=pd.DataFrame(),
        model=pd.DataFrame(),
        unit_conversion=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
        esm_db_name="esm_db",
        locations_ranking=["FR", "RER", "RoW"],
        main_database=Database(db_as_list=dummy_db),
        esm_location="FR",
    )

    updated_location = esm.change_location_activity(
        activity="Water",
        categories=('water',),
        location="GLO",
        database="biosphere3_regionalized_flows",
        technosphere_or_biosphere_db=Database(db_as_list=dummy_reg_biosphere_db),
        activity_type="biosphere",
    )

    assert updated_location == "RER"
