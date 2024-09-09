import pandas as pd
import bw2data as bd
import pytest
from mescal.modify_inventory import change_dac_biogenic_carbon_flow
from mescal.utils import write_wurst_database_to_brightway

dummy_esm_db = [
    {
        "name": "DAC_LT, Operation",
        "reference product": "carbon dioxide, captured from atmosphere",
        "location": "GLO",
        "database": "dummy_esm_db_dac",
        "unit": "kilogram",
        "code": "00001",
        "exchanges": [
            {
                "name": "DAC_LT, Operation",
                "product": "carbon dioxide, captured from atmosphere",
                "location": "GLO",
                "database": "dummy_esm_db_dac",
                "type": "production",
                "amount": 1,
                "unit": "kg",
                "code": "00001"
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
    }
]

tech_specifics = [
    ["DAC_LT, Operation", "DAC", ""],
]

tech_specifics = pd.DataFrame(tech_specifics, columns=["Name", "Specifics", "Amount"])


@pytest.mark.tags("requires_ecoinvent")
def test_change_dac_biogenic_carbon_flow():
    bd.projects.set_current('ecoinvent3.9.1')
    write_wurst_database_to_brightway(dummy_esm_db, "dummy_esm_db_dac")

    change_dac_biogenic_carbon_flow("dummy_esm_db_dac", "DAC_LT, Operation")

    act = [i for i in bd.Database("dummy_esm_db_dac").search("DAC_LT, Operation", limit=1000) if (
        ("DAC_LT, Operation" == i.as_dict()['name'])
    )][0]

    biosphere_flows = [i for i in act.biosphere()]

    assert len(biosphere_flows) == 1
    assert biosphere_flows[0].as_dict()['name'] == 'Carbon dioxide, fossil'
    assert biosphere_flows[0].as_dict()['categories'] == ('air',)
    assert biosphere_flows[0].as_dict()['input'] == ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
    assert biosphere_flows[0].as_dict()['amount'] == -1.0
