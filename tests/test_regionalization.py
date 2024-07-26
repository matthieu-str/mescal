from mescal.utils import database_list_to_dict
from mescal.regionalization import regionalize_activity_foreground
import copy
import pytest

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
    regionalized_act = regionalize_activity_foreground(
        act=act,
        target_region='FR',
        locations_ranking=['FR', 'RER', 'GLO', 'RoW'],
        accepted_locations=['FR'],
        db=dummy_db,
        db_dict_name=database_list_to_dict(dummy_db, 'name', 'technosphere'),
        db_dict_code=database_list_to_dict(dummy_db, 'code', 'technosphere'),
        regionalized_database=True,
        regionalized_biosphere_db=dummy_reg_biosphere_db,
        db_dict_name_reg_biosphere=database_list_to_dict(dummy_reg_biosphere_db, 'name', 'biosphere'),
    )

    for exc in regionalized_act['exchanges']:
        if exc['type'] == 'technosphere':
            assert (exc['location'] == 'FR') & (exc['input'][1] == '918932')
        elif exc['type'] == 'biosphere':
            assert (exc['name'] == 'Water, RER') & (exc['input'][1] == '0375893')
