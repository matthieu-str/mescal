import pandas as pd
import bw2data as bd
import pytest
from mescal.esm import ESM
from mescal.database import Database
from mescal.filesystem_constants import BW_PROJECT_NAME

dummy_esm_db = [
    {
        "name": "BATTERY, Construction",
        "reference product": "battery, Li-ion, LFP, rechargeable, prismatic",
        "location": "GLO",
        "database": "dummy_esm_db",
        "unit": "kilogram",
        "code": "00000",
        "exchanges": [
            {
                "name": "BATTERY, Construction",
                "product": "battery, Li-ion, LFP, rechargeable, prismatic",
                "location": "GLO",
                "database": "dummy_esm_db",
                "type": "production",
                "amount": 1,
                "unit": "kilogram",
                "code": "00000"
            },
            {
                "name": "market for battery, Li-ion, LFP, rechargeable, prismatic",
                "product": "battery, Li-ion, LFP, rechargeable, prismatic",
                "location": "GLO",
                "database": "ecoinvent-3.9.1-cutoff",
                "type": "technosphere",
                "amount": 1,
                "unit": "kilogram",
                "code": 'f22991a7a4e1d7ecfa81cdd7453e83d2',
            }
        ]
    },
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

mapping = [
    ['BATTERY', 'Construction', 'battery, Li-ion, LFP, rechargeable, prismatic',
     'market for battery, Li-ion, LFP, rechargeable, prismatic', 'GLO', 'ecoinvent-3.9.1-cutoff', '00000'],
    ['TRAIN_FREIGHT_DIESEL_LOC', 'Construction', 'locomotive', 'locomotive production', 'RER',
     'ecoinvent-3.9.1-cutoff', '00001'],
    ['TRAIN_FREIGHT_DIESEL_WAG', 'Construction', 'goods wagon', 'goods wagon production', 'RER',
     'ecoinvent-3.9.1-cutoff', '00002'],
]

technology_compositions = [
    ['TRAIN_FREIGHT_DIESEL', ['TRAIN_FREIGHT_DIESEL_LOC', 'TRAIN_FREIGHT_DIESEL_WAG']],
]

unit_conversion = [
    ['BATTERY', 'Construction', 10, 'kilogram', 'kilowatt hour'],
    ['TRAIN_FREIGHT_DIESEL_LOC', 'Construction', 2, 'unit', 'unit'],
    ['TRAIN_FREIGHT_DIESEL_WAG', 'Construction', 20, 'unit', 'unit'],
    ['TRAIN_FREIGHT_DIESEL', 'Construction', 2.5e-5, 'unit', 'ton kilometer per hour'],
]

lifetime = [
    ['BATTERY', 30, 20],
    ['TRAIN_FREIGHT_DIESEL', 40, None],
    ['TRAIN_FREIGHT_DIESEL_LOC', None, 50],
    ['TRAIN_FREIGHT_DIESEL_WAG', None, 50],
]

methods = ['IPCC 2021']

impact_abbrev = [
    ["('IPCC 2021', 'climate change', 'global warming potential (GWP100)')", "kg CO2-Eq", "CC", "CC"],
]

mapping = pd.DataFrame(mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database', 'New_code'])
technology_compositions = pd.DataFrame(technology_compositions, columns=['Name', 'Components'])
unit_conversion = pd.DataFrame(unit_conversion, columns=['Name', 'Type', 'Value', 'LCA', 'ESM'])
lifetime = pd.DataFrame(lifetime, columns=['Name', 'ESM', 'LCA'])
impact_abbrev = pd.DataFrame(impact_abbrev, columns=['Impact_category', 'Unit', 'Abbrev', 'AoP'])

def get_emissions_info(row):
    flow = bd.Database(row['database']).get(row['code'])
    return flow.as_dict()['name'], flow.as_dict()['categories']

@pytest.mark.tags("requires_ecoinvent")
def test_compute_impact_score():

    bd.projects.set_current(BW_PROJECT_NAME)
    Database(db_as_list=dummy_esm_db).write_to_brightway('dummy_esm_db')

    esm = ESM(
        mapping=mapping,
        technology_compositions=technology_compositions,
        unit_conversion=unit_conversion,
        lifetime=lifetime,
        main_database=Database('ecoinvent-3.9.1-cutoff'),
        esm_db_name='dummy_esm_db',
        model=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
    )

    R, df_contrib_results = esm.compute_impact_scores(
        methods=methods,
        impact_abbrev=impact_abbrev,
        specific_lcia_abbrev=['CC'],
        contribution_analysis="emissions",
    )

    df_contrib_results[['ef_name', 'ef_categories']] = pd.DataFrame(
        df_contrib_results.apply(lambda x: get_emissions_info(x), axis=1).tolist(),
        index=df_contrib_results.index
    )

    lcia_value_train = R[
        (R.Impact_category == ('IPCC 2021', 'climate change', 'global warming potential (GWP100)'))
        & (R.Name == 'TRAIN_FREIGHT_DIESEL')
        & (R.Type == 'Construction')
        ].Value.iloc[0]

    assert 44.40 <= lcia_value_train <= 44.41

    contrib_train = df_contrib_results[
        (df_contrib_results.impact_category == ('IPCC 2021', 'climate change', 'global warming potential (GWP100)'))
        & (df_contrib_results.act_name == 'TRAIN_FREIGHT_DIESEL')
        & (df_contrib_results.act_type == 'Construction')
    ]

    max_contrib_train = contrib_train[
        (contrib_train.score == contrib_train.score.max())
    ]

    assert max_contrib_train.ef_name.iloc[0] == 'Carbon dioxide, fossil'
    assert max_contrib_train.ef_categories.iloc[0] == ('air', 'non-urban air or from high stacks')
    assert 17.42 <= max_contrib_train.score.iloc[0] <= 17.43

    lcia_value_batt = R[
        (R.Impact_category == ('IPCC 2021', 'climate change', 'global warming potential (GWP100)'))
        & (R.Name == 'BATTERY')
        & (R.Type == 'Construction')
        ].Value.iloc[0]

    assert 167.60 <= lcia_value_batt <= 167.61

    contrib_batt = df_contrib_results[
        (df_contrib_results.impact_category == ('IPCC 2021', 'climate change', 'global warming potential (GWP100)'))
        & (df_contrib_results.act_name == 'BATTERY')
        & (df_contrib_results.act_type == 'Construction')
    ]

    max_contrib_batt = contrib_batt[
        (contrib_batt.score == contrib_batt.score.max())
    ]

    assert max_contrib_batt.ef_name.iloc[0] == 'Carbon dioxide, fossil'
    assert max_contrib_batt.ef_categories.iloc[0] == ('air', 'non-urban air or from high stacks')
    assert 98.78 <= max_contrib_batt.score.iloc[0] <= 98.79
