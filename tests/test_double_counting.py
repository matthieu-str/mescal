import pandas as pd
import pytest
from mescal.database import Database, Dataset
from mescal.esm import ESM

dummy_db = [
    {
        "name": "market for heat pump production",
        "reference product": "heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00000",
        "classifications": [("CPC", "1234: Construction")],
        "exchanges": [
            {
                "name": "market for heat pump production",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00000",
            },
            {
                "name": "heat pump production (type I)",
                "product": "heat pump",
                "unit": "unit",
                "amount": 0.4,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00001",
            },
            {
                "name": "heat pump production (type II)",
                "product": "heat pump",
                "unit": "unit",
                "amount": 0.6,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00002",
            }
        ]
    },
    {
        "name": "heat pump production (type I)",
        "reference product": "heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00001",
        "classifications": [("CPC", "1234: Construction")],
        "exchanges": [
            {
                "name": "heat pump production (type I)",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00001",
            },
            {
                "name": "steel production",
                "product": "steel",
                "unit": "kilogram",
                "amount": 50,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00003",
            },
            {
                "name": "steel waste, recycled",
                "product": "waste steel",
                "unit": "kilogram",
                "amount": -50,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00004",
            },
        ]
    },
    {
        "name": "heat pump production (type II)",
        "reference product": "heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00002",
        "classifications": [("CPC", "1234: Construction")],
        "exchanges": [
            {
                "name": "heat pump production (type II)",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00002",
            },
            {
                "name": "copper production",
                "product": "copper",
                "unit": "kilogram",
                "amount": 20,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00005",
            },
            {
                "name": "copper waste, recycled",
                "product": "waste copper",
                "unit": "kilogram",
                "amount": -20,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00006",
            },
        ]
    },
    {
        "name": "steel production",
        "reference product": "steel",
        "location": "CH",
        "unit": "kilogram",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00003",
        "classifications": [("CPC", "5678: Steel")],
        "exchanges": [
            {
                "name": "steel production",
                "product": "steel",
                "unit": "kilogram",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00003",
            }
        ]
    },
    {
        "name": "copper production",
        "reference product": "copper",
        "location": "CH",
        "unit": "kilogram",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00005",
        "classifications": [("CPC", "6789: Copper")],
        "exchanges": [
            {
                "name": "copper production",
                "product": "copper",
                "unit": "kilogram",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00005",
            }
        ]
    },
    {
        "name": "steel waste, recycled",
        "reference product": "waste steel",
        "location": "CH",
        "unit": "kilogram",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00004",
        "classifications": [("CPC", "9876: Waste")],
        "exchanges": [
            {
                "name": "steel waste, recycled",
                "product": "waste steel",
                "unit": "kilogram",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00004",
            }
        ]
    },
    {
        "name": "copper waste, recycled",
        "reference product": "waste copper",
        "location": "CH",
        "unit": "kilogram",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00006",
        "classifications": [("CPC", "9876: Waste")],
        "exchanges": [
            {
                "name": "copper waste, recycled",
                "product": "waste copper",
                "unit": "kilogram",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00006",
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
                "amount": 0.01,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "a44444",
            },
            {
                "name": "refrigerant R134a production",
                "product": "refrigerant R134a",
                "unit": "cubic meter",
                "amount": 3e-6,
                "type": "technosphere",
                "location": "CH",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "55555",
            },
            {
                "name": "market for heat pump production",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1e-6,
                "type": "technosphere",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "00000",
            }
        ]
    },
    {
        "name": "market for electricity, low voltage",
        "reference product": "electricity, low voltage",
        "location": "CH",
        "unit": "kilowatt hour",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "a44444",
        "classifications": [("CPC", "1000: Electricity")],
        "exchanges": [
            {
                "name": "market for electricity, low voltage",
                "product": "electricity, low voltage",
                "unit": "kilowatt hour",
                "amount": 1,
                "type": "production",
                "database": "ecoinvent-3.9.1-cutoff",
                "code": "a44444",
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
    }
]

mapping = [
    ['HEAT_PUMP', 'Operation', 'heat', 'market for heat production by heat pump', 'CH', '/megajoule',
     'ecoinvent-3.9.1-cutoff'],
    ['HEAT_PUMP', 'Construction', 'heat pump', 'market for heat pump production', 'CH', '/unit', 'ecoinvent-3.9.1-cutoff']
]

mapping_esm_flows_to_CPC = [
    ['ELECTRICITY', ['1000: Electricity']],
    ['HEAT', ['2000: Heat']],
    ['TRANSPORT', ['5000: Transport']],
    ['CONSTRUCTION', ['1234: Construction']],
    ['REFRIGERANTS', ['9999: Refrigerants']],
    ['DECOMMISSION', ['9876: Waste']],
]

model = [
    ['HEAT_PUMP', 'HEAT', 1],
    ['HEAT_PUMP', 'ELECTRICITY', -0.25],
    ['HEAT_PUMP', 'CONSTRUCTION', -1],
]

unit_conversion = [
    ['HEAT_PUMP', 'Operation', 'megajoule', 'kilowatt hour', 3.6],
    ['HEAT_PUMP', 'Construction', 'unit', 'kilowatt', 1/30],
]

mapping = pd.DataFrame(data=mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Unit', 'Database'])
mapping_esm_flows_to_CPC = pd.DataFrame(mapping_esm_flows_to_CPC, columns=['Flow', 'CPC'])
model = pd.DataFrame(model, columns=['Name', 'Flow', 'Amount'])
unit_conversion = pd.DataFrame(data=unit_conversion, columns=['Name', 'Type', 'LCA', 'ESM', 'Value'])


@pytest.mark.tags("workflow")
def test_create_esm_database():

    esm = ESM(
        mapping=mapping,
        model=model,
        mapping_esm_flows_to_CPC_cat=mapping_esm_flows_to_CPC,
        esm_db_name='esm_db_name',
        main_database=Database(db_as_list=dummy_db),
        unit_conversion=unit_conversion,
        extract_eol_from_construction=True,
        remove_double_counting_to=['Operation', 'Construction'],
    )

    esm_db = esm.create_esm_database(
        write_database=False,
        return_database=True,
        write_double_counting_removal_reports=False,
    )

    esm_db_dict_name = esm_db.db_as_dict_name
    act_op = esm_db_dict_name["HEAT_PUMP, Operation", "heat", "CH", 'esm_db_name']
    act_op_technosphere = Dataset(act_op).get_technosphere_flows()

    assert ((act_op_technosphere[0]['amount'] == 1)  # this exchange has to be kept
            & (act_op_technosphere[0]['database'] == 'esm_db_name'))
    # this activity should be copied in the ESM database

    act = esm_db_dict_name['heat production by heat pump', 'heat', 'CH', 'esm_db_name']
    act_technosphere = Dataset(act).get_technosphere_flows()

    for exc in act_technosphere:
        if exc['name'] == 'market for electricity, low voltage':
            assert exc['amount'] == 0  # electricity flow has to be removed
        elif exc['name'] == 'refrigerant R134a production':
            assert exc['amount'] == 3e-6  # refrigerant flow has to be kept
        elif exc['name'] == 'HEAT_PUMP, Construction':
            assert exc['amount'] == 0  # construction flow has to be removed
        else:
            raise ValueError(f"Unexpected technosphere exchange: {exc}")

    act_constr = esm_db_dict_name["heat pump production (type I)", "heat pump", "CH", "esm_db_name"]
    act_constr_technosphere = Dataset(act_constr).get_technosphere_flows()

    for exc in act_constr_technosphere:
        if exc['name'] == 'steel production':
            assert exc['amount'] == 50  # steel production flow has to be kept
        elif exc['name'] == 'steel waste, recycled':
            assert exc['amount'] == 0  # steel waste flow has to be removed (moved to the decommission dataset)
        else:
            raise ValueError(f"Unexpected technosphere exchange: {exc}")

    act_decom = esm_db_dict_name["HEAT_PUMP, Decommission", "used heat pump", "CH", "esm_db_name"]
    act_decom_technosphere = Dataset(act_decom).get_technosphere_flows()

    for exc in act_decom_technosphere:
        if exc['name'] == 'steel waste, recycled':
            assert exc['amount'] == -50 * 0.4  # steel waste flow scaled to the functional unit
        elif exc['name'] == 'copper waste, recycled':
            assert exc['amount'] == -20 * 0.6  # copper waste flow scaled to the functional unit
        else:
            raise ValueError(f"Unexpected technosphere exchange: {exc}")