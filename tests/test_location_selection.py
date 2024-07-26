from mescal.location_selection import change_location_activity
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
    updated_location = change_location_activity(
        activity="market for electricity, low voltage",
        product="electricity, low voltage",
        location="DE",
        database="ecoinvent-3.9.1-cutoff",
        locations_ranking=["FR", "RER", "RoW"],
        db=dummy_db,
        esm_region="FR",
        activity_type="technosphere",
    )
    assert updated_location == "FR"


@pytest.mark.tags("workflow")
def test_change_location_activity_biosphere():
    updated_location = change_location_activity(
        activity="Water",
        categories=('water',),
        location="GLO",
        database="biosphere3_regionalized_flows",
        locations_ranking=["RER", "CH"],
        db=dummy_reg_biosphere_db,
        esm_region="RER",
        activity_type="biosphere",
    )
    assert updated_location == "RER"
