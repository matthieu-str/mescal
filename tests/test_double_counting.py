import pandas as pd
from mescal.utils import database_list_to_dict, get_technosphere_flows
from mescal.double_counting import create_esm_database

dummy_db = [
    {
        "name": "heat pump production",
        "reference product": "heat pump",
        "location": "CH",
        "unit": "unit",
        "database": "ecoinvent-3.9.1-cutoff",
        "code": "00000",
        "classifications": [("CPC", "1234: Construction")],
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
                "name": "heat pump production",
                "product": "heat pump",
                "unit": "unit",
                "amount": 1e-8,
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
        "code": "44444",
        "classifications": [("CPC", "1000: Electricity")],
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
    ['HEAT_PUMP', 'Construction', 'heat pump', 'heat pump production', 'CH', '/unit', 'ecoinvent-3.9.1-cutoff']
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
]

mapping = pd.DataFrame(data=mapping, columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Unit', 'Database'])
mapping_esm_flows_to_CPC = pd.DataFrame(mapping_esm_flows_to_CPC, columns=['Flow', 'CPC'])
model = pd.DataFrame(model, columns=['Name', 'Flow', 'Amount'])


def test_create_esm_database():

    esm_db = create_esm_database(
        mapping=mapping,
        model=model,
        mapping_esm_flows_to_CPC_cat=mapping_esm_flows_to_CPC,
        esm_db_name='esm_db_name',
        main_database=dummy_db,
        write_database=False,
        return_obj='database',
    )

    esm_db_dict_name = database_list_to_dict(esm_db, "name")
    act_op = esm_db_dict_name["HEAT_PUMP, Operation", "heat", "CH", 'esm_db_name']
    act_op_technosphere = get_technosphere_flows(act_op)

    assert ((act_op_technosphere[0]['amount'] == 1)  # this exchange has to be kept
            & (act_op_technosphere[0]['database'] == 'esm_db_name'))
    # this activity should be copied in the ESM database

    act = esm_db_dict_name['heat production by heat pump', 'heat', 'CH', 'esm_db_name']
    act_technosphere = get_technosphere_flows(act)

    for exc in act_technosphere:
        if exc['name'] == 'market for electricity, low voltage':
            assert exc['amount'] == 0  # electricity flow has to be removed
        elif exc['name'] == 'refrigerant R134a production':
            assert exc['amount'] == 3e-2  # refrigerant flow has to be kept
        elif exc['name'] == 'heat pump production':
            assert exc['amount'] == 0  # construction flow has to be removed
        else:
            raise ValueError("Unexpected technosphere exchange")