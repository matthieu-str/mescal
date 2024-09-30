import pandas as pd
import pytest
from mescal.esm import ESM
from mescal.database import Database, Dataset

dummy_db = [
    {
        "name": "heat pump production",
        "reference product": "heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00000",
        "classifications": [("CPC", "1234: Construction")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat pump production",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00000",
            }
        ]
    },
    {
        "name": "double-stage heat pump production",
        "reference product": "double-stage heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "11000",
        "classifications": [("CPC", "1234: Construction")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat pump production",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00000",
            }
        ]
    },
    {
        "name": "market for heat production by heat pump",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "99999",
        "classifications": [("CPC", "3000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "market for heat production by heat pump",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "99999",
            },
            {
                "name": "heat production by heat pump",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "33333",
            }
        ]
    },
    {
        "name": "heat production by heat pump",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "33333",
        "classifications": [("CPC", "3000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat production by heat pump",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "33333",
            },
            {
                "name": "market for electricity, low voltage",
                "product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "amount": 0.05,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "44444",
            },
            {
                "name": "refrigerant R134a production",
                "product": "refrigerant R134a",
                "unit": "cubic meter",
                "amount": 3e-2,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "55555",
            },
            {
                "name": "double-stage heat pump production",
                "product": "double-stage heat pump",
                "unit": "unit",
                "amount": 1e-8,
                "type": "technosphere",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "11000",
            }
        ]
    },
    {
        "name": "heat production by double-stage heat pump",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "88888",
        "classifications": [("CPC", "3000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat production by double-stage heat pump",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "88888",
            },
            {
                "name": "market for electricity, low voltage",
                "product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "amount": 0.04,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "44444",
            },
            {
                "name": "refrigerant R134a production",
                "product": "refrigerant R134a",
                "unit": "cubic meter",
                "amount": 3e-2,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "55555",
            },
            {
                "name": "double-stage heat pump production",
                "product": "double-stage heat pump",
                "unit": "unit",
                "amount": 1e-8,
                "type": "technosphere",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "11000",
            }
        ]
    },
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "CH",
        "unit": "kilowatt hour",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "44444",
        "classifications": [("CPC", "1000: Electricity")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "market for electricity, low voltage",
                "product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "44444",
            }
        ]
    },
    {
        "name": "refrigerant R134a production",
        "reference product": "refrigerant R134a",
        "location": "CH",
        "unit": "cubic meter",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "55555",
        "classifications": [("CPC", "9999: Refrigerants")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "refrigerant R134a production",
                "product": "refrigerant R134a",
                "unit": "cubic meter",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "55555",
            }
        ]
    },
    {
        "name": "market for heat",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "54321",
        "classifications": [("CPC", "2000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "market for heat",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "54321",
            },
            {
                "name": "heat production by direct electric heater",
                "product": "heat",
                "unit": "megajoule",
                "amount": 0.5,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00011",
            },
            {
                "name": "heat production, from oil boiler",
                "product": "heat",
                "unit": "megajoule",
                "amount": 0.25,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00022",
            },
            {
                "name": "heat production, from natural gas boiler",
                "product": "heat",
                "unit": "megajoule",
                "amount": 0.25,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00033",
            }
        ]
    },
    {
        "name": "heat production by direct electric heater",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00011",
        "classifications": [("CPC", "3000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat production by direct electric heater",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00011",
            }
        ]
    },
    {
        "name": "heat production, from oil boiler",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00022",
        "classifications": [("CPC", "3000: Heat")],
        "exchanges": [
            {
                "name": "heat production, from oil boiler",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00022",
            }
        ]
    },
    {
        "name": "heat production, from natural gas boiler",
        "reference product": "heat",
        "location": "CH",
        "unit": "megajoule",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00033",
        "classifications": [("CPC", "3000: Heat")],
        "comment": "dummy comment",
        "exchanges": [
            {
                "name": "heat production, from natural gas boiler",
                "product": "heat",
                "unit": "megajoule",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00033",
            }
        ]
    }
]

mapping = [
    ['HEAT_PUMP', 'Operation', 'heat', 'market for heat production by heat pump', 'CH', '/megajoule',
     'ecoinvent-3.9.1-cutoff'],
    ['HEAT_PUMP', 'Construction', 'heat pump', 'heat pump production', 'CH', '/unit', 'ecoinvent-3.9.1-cutoff'],
    ['HEAT_PUMP_DOUBLE_STAGE', 'Construction', 'double-stage heat pump', 'double-stage heat pump production', 'CH',
     '/unit', 'ecoinvent-3.9.1-cutoff'],
    ['HEAT_PUMP_DOUBLE_STAGE', 'Operation', 'heat', 'heat production by double-stage heat pump', 'CH',
     '/megajoule', 'ecoinvent-3.9.1-cutoff'],
    ['HEAT', 'Flow', 'heat', 'market for heat', 'CH', '/megajoule', 'ecoinvent-3.9.1-cutoff'],
]

mapping_esm_flows_to_CPC = [
    ['ELECTRICITY', ['1000: Electricity']],
    ['HEAT', ['2000: Heat']],
    ['TRANSPORT', ['5000: Transport']],
    ['CONSTRUCTION', ['1234: Construction']],
    ['REFRIGERANTS', ['9999: Refrigerants']],
]

model = [
    ['HEAT_PUMP', 'HEAT', 1],
    ['HEAT_PUMP', 'ELECTRICITY', -0.25],
    ['HEAT_PUMP', 'CONSTRUCTION', -1],
    ['HEAT_PUMP_DOUBLE_STAGE', 'HEAT', 1],
    ['HEAT_PUMP_DOUBLE_STAGE', 'ELECTRICITY', -0.20],
    ['HEAT_PUMP_DOUBLE_STAGE', 'CONSTRUCTION', -1],
]

esm_results = [
    ['HEAT_PUMP', 10],
    ['HEAT_PUMP_DOUBLE_STAGE', 25],
]

unit_conversion = [
    ['HEAT_PUMP', 'Operation', 3.6, 'megajoule', 'kilowatt hour'],
    ['HEAT_PUMP', 'Construction', 1/5, 'unit', 'kilowatt'],
    ['HEAT_PUMP_DOUBLE_STAGE', 'Operation', 3.6, 'megajoule', 'kilowatt hour'],
    ['HEAT_PUMP_DOUBLE_STAGE', 'Construction', 1/10, 'unit', 'kilowatt'],
]

mapping = pd.DataFrame(data=mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Unit', 'Database'])
mapping_esm_flows_to_CPC = pd.DataFrame(mapping_esm_flows_to_CPC, columns=['Flow', 'CPC'])
model = pd.DataFrame(model, columns=['Name', 'Flow', 'Amount'])
esm_results = pd.DataFrame(esm_results, columns=['Name', 'Production'])
unit_conversion = pd.DataFrame(unit_conversion, columns=['Name', 'Type', 'Value', 'LCA', 'ESM'])


@pytest.mark.tags("workflow")
def test_create_new_database_with_esm_results():

    esm = ESM(
        mapping=mapping,
        model=model,
        unit_conversion=unit_conversion,
        mapping_esm_flows_to_CPC_cat=mapping_esm_flows_to_CPC,
        esm_db_name="esm_db",
        esm_location="CH",
        locations_ranking=["CH", "RER", "GLO"],
        accepted_locations=["CH"],
        main_database=Database(db_as_list=dummy_db),
    )

    esm.create_new_database_with_esm_results(
        esm_results=esm_results,
        write_database=False,
    )

    new_db_dict_name = esm.main_database.db_as_dict_name

    new_act = new_db_dict_name[
        "market for heat, from ESM results", "heat", "CH", 'ecoinvent-3.9.1-cutoff_with_ESM_results_for_CH'
    ]
    new_act_technosphere_flows = Dataset(new_act).get_technosphere_flows()

    # Test the entries of the new dataset
    assert len(new_act_technosphere_flows) == 2  # single and double-stage HPs flows
    assert (((new_act_technosphere_flows[0]['name'] == 'market for heat production by heat pump')
            & (new_act_technosphere_flows[1]['name'] == "heat production by double-stage heat pump"))
            | ((new_act_technosphere_flows[1]['name'] == 'market for heat production by heat pump')
            & (new_act_technosphere_flows[0]['name'] == "heat production by double-stage heat pump")))
    if new_act_technosphere_flows[0]['name'] == 'market for heat production by heat pump':
        assert new_act_technosphere_flows[0]['amount'] == 10 / (10 + 25)
        assert new_act_technosphere_flows[1]['amount'] == 25 / (10 + 25)
    else:
        assert new_act_technosphere_flows[1]['amount'] == 10 / (10 + 25)
        assert new_act_technosphere_flows[0]['amount'] == 25 / (10 + 25)

    # Test that double-counting removal on the construction flows has been done
    hp_act = new_db_dict_name[
        "market for heat production by heat pump", "heat", "CH", 'ecoinvent-3.9.1-cutoff_with_ESM_results_for_CH'
    ]
    hp_act_technosphere_flows = Dataset(hp_act).get_technosphere_flows()
    for exc in hp_act_technosphere_flows:
        if exc['name'] == 'heat pump production':
            assert exc['amount'] == 0

    double_stage_hp_act = new_db_dict_name[
        "heat production by double-stage heat pump", "heat", "CH", 'ecoinvent-3.9.1-cutoff_with_ESM_results_for_CH'
    ]
    double_stage_hp_act_technosphere_flows = Dataset(double_stage_hp_act).get_technosphere_flows()
    for exc in double_stage_hp_act_technosphere_flows:
        if exc['name'] == 'double-stage heat pump production':
            assert exc['amount'] == 0
