from mescal.esm import ESM
from mescal.database import Database
import copy
import pytest
import pandas as pd

dummy_db = [
    {
        'name': 'fake activity',
        'location': 'CH',
        'reference product': 'fake product',
        'unit': "kilogram",
        'database': 'ecoinvent-3.9.1-cutoff',
        'code': '123456',
        'exchanges': [
            {
                "name": "fake activity",
                "product": "fake product",
                "amount": 1,
                "type": "production",
                "unit": "kilogram",
                "database": "fake database",
                "code": "123456",
                "input": ("fake database", "123456"),
            },
            {
                "name": "Water, CH",
                "categories": ('water',),
                "amount": 1.5,
                "type": "biosphere",
                "unit": "cubic meter",
                "database": "biosphere3_regionalized_flows",
                "code": "876473",
                "input": ("biosphere3_regionalized_flows", "876473")
            },
            {
                "name": "market for electricity, low voltage",
                "product": "electricity, low voltage",
                "amount": 20,
                "location": "CH",
                "type": "technosphere",
                "unit": "kilowatt hour",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "876473",
                "input": ("ecoinvent-3.9.1-cutoff", "876473")
            },
        ]
    },
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "CH",
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
        "code": "0375893",
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
def test_regionalize_activity_foreground():
    act = copy.deepcopy(dummy_db[0])

    esm = ESM(
        mapping=pd.DataFrame(),
        model=pd.DataFrame(),
        unit_conversion=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
        esm_db_name="esm_db",
        locations_ranking=['FR', 'RER', 'GLO', 'RoW'],
        accepted_locations=['FR'],
        esm_location="FR",
        spatialized_biosphere_db=Database(db_as_list=dummy_reg_biosphere_db),
        main_database=Database(db_as_list=dummy_db),
    )
    esm.best_loc_in_ranking = {}  # reset the best location in ranking (possibly set by previous tests)
    regionalized_act = esm.regionalize_activity_foreground(act=act)

    for exc in regionalized_act['exchanges']:
        if exc['type'] == 'technosphere':
            assert (exc['location'] == 'FR') & (exc['input'][1] == '918932')
        elif exc['type'] == 'biosphere':
            assert (exc['name'] == 'Water, RER') & (exc['input'][1] == '0375893')
