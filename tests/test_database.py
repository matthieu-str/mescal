import pandas as pd
from mescal.database import Database, Dataset
from mescal.filesystem_constants import BW_PROJECT_NAME
import bw2data as bd
import pytest

dummy_ds = {
    'name': 'dummy activity',
    'reference product': 'dummy product',
    'unit': 'kg',
    'location': 'GLO',
    'database': 'dummy database',
    'code': '00000',
    'exchanges': [
        {
            'name': 'A',
            'amount': 1,
            'unit': 'kg',
            'type': 'production',
            'location': 'GLO',
            'code': '00A',
        },
        {
            'name': 'B',
            'amount': 2,
            'unit': 'kg',
            'type': 'technosphere',
            'location': 'GLO',
            'code': '00B',
        },
        {
            'name': 'C',
            'amount': 3,
            'unit': 'kg',
            'type': 'biosphere',
            'location': 'GLO',
            'categories': ('C',),
            'code': '00C',
        },
        {
            'name': 'D',
            'amount': 4,
            'unit': 'kg',
            'type': 'biosphere',
            'location': 'GLO',
            'categories': ('D',),
            'code': '00D',
        },
    ],
}

dummy_esm_db = [
    {
        "name": "TRAIN_FREIGHT_DIESEL_LOC, Construction",
        "reference product": "locomotive",
        "location": "RER",
        "database": "dummy_esm_db",
        "unit": "unit",
        "code": "00001",
        "exchanges": [
            {
                "name": "TRAIN_FREIGHT_DIESEL_LOC, Construction",
                "product": "locomotive",
                "location": "RER",
                "database": "dummy_esm_db",
                "type": "production",
                "amount": 1,
                "unit": "unit",
                "code": "00001"
            },
            {
                "name": "locomotive production",
                "product": "locomotive",
                "location": "RER",
                "database": "ecoinvent-3.9.1-cutoff",
                "type": "technosphere",
                "amount": 1,
                "unit": "unit",
                "code": '54c4d94036d1e4d8e930bbe55332f066',
            }
        ],
    },
    {
        "name": "TRAIN_FREIGHT_DIESEL_WAG, Construction",
        "reference product": "goods wagon",
        "location": "RER",
        "database": "dummy_esm_db",
        "unit": "unit",
        "code": "00002",
        "exchanges": [
            {
                "name": "TRAIN_FREIGHT_DIESEL_WAG, Construction",
                "product": "goods wagon",
                "location": "RER",
                "database": "dummy_esm_db",
                "type": "production",
                "amount": 1,
                "unit": "unit",
                "code": "00002"
            },
            {
                "name": "goods wagon production",
                "product": "goods wagon",
                "location": "RER",
                "database": "ecoinvent-3.9.1-cutoff",
                "type": "technosphere",
                "amount": 1,
                "unit": "unit",
                "code": '4b775f75ed40167082fd41f85e19e978',
            }
        ],
    }
]

dummy_esm_db_2 = [
    {
        "name": "DAC_LT, Operation",
        "reference product": "carbon dioxide, captured from atmosphere",
        "location": "GLO",
        "database": "dummy_esm_db_dac",
        "unit": "kilogram",
        "code": "00003",
        "exchanges": [
            {
                "name": "DAC_LT, Operation",
                "product": "carbon dioxide, captured from atmosphere",
                "location": "GLO",
                "database": "dummy_esm_db_dac",
                "type": "production",
                "amount": 1,
                "unit": "kg",
                "code": "00003"
            },
            {
                "name": "heat production, natural gas, at industrial furnace >100kW",
                "product": "heat, district or industrial, natural gas",
                "location": "RoW",
                "database": "ecoinvent-3.9.1-cutoff",
                "type": "technosphere",
                "amount": 5.4,
                "unit": "megajoule",
                "code": "586243df1b3c97d5fea56c408f3fcdef",
                "input": ("ecoinvent-3.9.1-cutoff", "586243df1b3c97d5fea56c408f3fcdef")
            },
            {
                "name": "Carbon dioxide, non-fossil",
                "categories": ("natural resource", "in air"),
                "database": "biosphere3",
                "input": ("biosphere3", "eba59fd6-f37e-41dc-9ca3-c7ea22d602c7"),
                "amount": 1,
                "type": "biosphere",
                "unit": "kilogram",
            }
        ],
    },
    {
        "name": "CAR_BIODIESEL, Operation",
        "reference product": "transport, passenger car",
        "location": "GLO",
        "database": "dummy_esm_db_dac",
        "unit": "kilometer",
        "code": "00004",
        "exchanges": [
            {
                "name": "CAR_BIODIESEL, Operation",
                "product": "transport, passenger car",
                "location": "GLO",
                "database": "dummy_esm_db_dac",
                "type": "production",
                "amount": 1,
                "unit": "km",
                "code": "00004"
            },
            {
                "name": "market for passenger car, diesel",
                "product": "passenger car, diesel",
                "location": "GLO",
                "database": "ecoinvent-3.9.1-cutoff",
                "type": "technosphere",
                "amount": 5e-6,
                "unit": "unit",
                "code": "94ce79023f7075e5579a8e2984d5059e",
                "input": ('ecoinvent-3.9.1-cutoff', '94ce79023f7075e5579a8e2984d5059e')
            },
            {
                "name": "Carbon dioxide, fossil",
                "categories": ("air",),
                "database": "biosphere3",
                "input": ("biosphere3", "349b29d1-3e58-4c66-98b9-9d1a076efd2e"),
                "amount": 0.20,
                "type": "biosphere",
                "unit": "kilogram",
            },
            {
                "name": "Carbon dioxide, fossil",
                "categories": ('air', 'urban air close to ground'),
                "database": "biosphere3",
                "input": ('biosphere3', 'f9749677-9c9f-4678-ab55-c607dfdc2cb9'),
                "amount": 0.30,
                "type": "biosphere",
                "unit": "kilogram",
            },
            {
                "name": "Carbon monoxide, fossil",
                "categories": ("air",),
                "database": "biosphere3",
                "input": ('biosphere3', 'ba2f3f82-c93a-47a5-822a-37ec97495275'),
                "amount": 0.10,
                "type": "biosphere",
                "unit": "kilogram",
            }
        ],
    }
]

mapping = [
    ['Tech 1', 'Operation', 'carbon dioxide, captured from atmosphere', 'DAC_LT, Operation', 'GLO', 'dummy_esm_db_dac'],
    ['Tech 2', 'Operation', 'transport, passenger car', 'CAR_BIODIESEL, Operation', 'GLO', 'dummy_esm_db_dac'],
    ['Tech 3', 'Construction', 'locomotive', 'TRAIN_FREIGHT_DIESEL_LOC, Construction', 'RER', 'dummy_esm_db'],
]

mapping_product_to_CPC = [
    ['locomotive', '0001: Locomotives', 'equals', 'Product'],
    ['goods wagon', '0002: Railway or tramway goods', 'equals', 'Product'],
]

mapping = pd.DataFrame(data=mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database'])
mapping_product_to_CPC = pd.DataFrame(data=mapping_product_to_CPC, columns=['Name', 'CPC', 'Search type', 'Where'])


@pytest.mark.tags("requires_ecoinvent")
def test_database():
    bd.projects.set_current(BW_PROJECT_NAME)

    multiple_dbs = Database(db_names=["ecoinvent-3.9.1-cutoff", "ecoinvent_cutoff_3.9.1_image_SSP2-Base_2050"])
    assert len(multiple_dbs.db_as_list) == 21238 + 32527
    assert set([x['database'] for x in multiple_dbs.db_as_list]) == {'ecoinvent-3.9.1-cutoff',
                                                                     'ecoinvent_cutoff_3.9.1_image_SSP2-Base_2050'}

    new_db = Database(db_as_list=dummy_esm_db+dummy_esm_db_2)
    new_db.relink(  # test the relink method
        name_database_unlink='ecoinvent-3.9.1-cutoff',
        database_relink_as_list=Database('ecoinvent_cutoff_3.9.1_image_SSP2-Base_2050').db_as_list,
        name_new_db='new_dummy_esm_db',  # testing the change_name method
        write=True,  # testing the write_to_brightway method
    )

    new_db = Database(db_names='new_dummy_esm_db')
    assert len(new_db.db_as_list) == 4
    assert new_db.db_as_list[0]['database'] == 'new_dummy_esm_db'
    dependencies = new_db.dependencies()
    assert 'ecoinvent-3.9.1-cutoff' not in dependencies
    assert 'ecoinvent_cutoff_3.9.1_image_SSP2-Base_2050' in dependencies


@pytest.mark.tags("workflow")
def test_CPC():
    db = Database(db_as_list=dummy_esm_db)
    db.add_CPC_categories(
        mapping_product_to_CPC=mapping_product_to_CPC,
    )
    if db.db_as_list[0]['name'] == 'TRAIN_FREIGHT_DIESEL_LOC, Construction':
        assert dict(db.db_as_list[0]['classifications'])['CPC'] == '0001: Locomotives'
        assert dict(db.db_as_list[1]['classifications'])['CPC'] == '0002: Railway or tramway goods'
    else:
        assert dict(db.db_as_list[0]['classifications'])['CPC'] == '0002: Railway or tramway goods'
        assert dict(db.db_as_list[1]['classifications'])['CPC'] == '0001: Locomotives'


@pytest.mark.tags("requires_ecoinvent")
def test_complementary_database():
    bd.projects.set_current(BW_PROJECT_NAME)
    db = Database(db_as_list=dummy_esm_db)
    db.write_to_brightway('dummy_esm_db')
    db2 = Database(db_as_list=dummy_esm_db_2)
    db2.write_to_brightway('dummy_esm_db_dac')

    main_db = Database(db_names='ecoinvent-3.9.1-cutoff')
    main_db.create_complementary_database(
        df_mapping=mapping,
        main_db_name='ecoinvent-3.9.1-cutoff',
        complement_db_name='dummy_complement_db',
    )

    complement_db = Database(db_names='dummy_complement_db')
    assert len(complement_db.db_as_list) == 3
    assert complement_db.db_as_list[0]['database'] == 'dummy_complement_db'
    assert set([i['name'] for i in complement_db.db_as_list]) == {'DAC_LT, Operation', 'CAR_BIODIESEL, Operation',
                                                                  'TRAIN_FREIGHT_DIESEL_LOC, Construction'}


@pytest.mark.tags("workflow")
def test_dataset():
    ds = Dataset(dummy_ds)
    assert len(ds.get_technosphere_flows()) == 1
    assert ds.get_technosphere_flows()[0]['name'] == 'B'
    assert len(ds.get_biosphere_flows()) == 2


@pytest.mark.tags("workflow")
def test_add_sub_databases():
    db1 = Database(db_as_list=dummy_esm_db)
    db2 = Database(db_as_list=dummy_esm_db_2)
    db = db1 + db2
    assert len(db) == 4

    db = db - db2
    assert len(db) == 2
    for act in db1.db_as_list:
        assert act in db.db_as_list
