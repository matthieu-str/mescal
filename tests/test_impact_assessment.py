import pandas as pd
import bw2data as bd
import pytest
from mescal.impact_assessment import compute_impact_scores
from mescal.utils import write_wurst_database_to_brightway

# Load ecoinvent
bd.projects.set_current('ecoinvent3.9.1')

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

write_wurst_database_to_brightway(dummy_esm_db, 'dummy_esm_db')

mapping = [
    ['TRAIN_FREIGHT_DIESEL_LOC', 'Construction', 'locomotive', 'locomotive production', 'RER',
     'ecoinvent-3.9.1-cutoff', '00001'],
    ['TRAIN_FREIGHT_DIESEL_WAG', 'Construction', 'goods wagon', 'goods wagon production', 'RER',
     'ecoinvent-3.9.1-cutoff', '00002'],
]

technology_compositions = [
    ['TRAIN_FREIGHT_DIESEL', ['TRAIN_FREIGHT_DIESEL_LOC', 'TRAIN_FREIGHT_DIESEL_WAG']],
]

methods = ['IMPACT World+ Damage 2.0.1_regionalized']

unit_conversion = [
    ['TRAIN_FREIGHT_DIESEL_LOC', 'Construction', 2, 'unit', 'unit'],
    ['TRAIN_FREIGHT_DIESEL_WAG', 'Construction', 20, 'unit', 'unit'],
    ['TRAIN_FREIGHT_DIESEL', 'Construction', 2.5e-5, 'unit', 'ton kilometer per hour'],
]

lifetime = [
    ['TRAIN_FREIGHT_DIESEL', 40, None],
    ['TRAIN_FREIGHT_DIESEL_LOC', None, 50],
    ['TRAIN_FREIGHT_DIESEL_WAG', None, 50],
]

mapping = pd.DataFrame(mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database', 'New_code'])
technology_compositions = pd.DataFrame(technology_compositions, columns=['Name', 'Components'])
unit_conversion = pd.DataFrame(unit_conversion, columns=['Name', 'Type', 'Value', 'LCA', 'ESM'])
lifetime = pd.DataFrame(lifetime, columns=['Name', 'ESM', 'LCA'])


@pytest.mark.tags("requires_ecoinvent")
def test_compute_impact_score():
    R = compute_impact_scores(
        esm_db=dummy_esm_db,
        mapping=mapping,
        technology_compositions=technology_compositions,
        unit_conversion=unit_conversion,
        lifetime=lifetime,
        methods=methods,
    )

    lcia_value = R[
        (R.Impact_category == ('IMPACT World+ Damage 2.0.1_regionalized',
                               'Ecosystem quality', 'Total ecosystem quality'))
        & (R.Name == 'TRAIN_FREIGHT_DIESEL')
        & (R.Type == 'Construction')
        ].Value.iloc[0]

    assert 58.44 <= lcia_value <= 58.45
