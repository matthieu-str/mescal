import pandas as pd
import pytest
from mescal.adapt_efficiency import correct_esm_and_lca_efficiency_differences
from mescal.utils import database_list_to_dict, get_biosphere_flows

dummy_db = [
    {
        'name': 'CAR_GASOLINE, Operation',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00000',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'CAR_GASOLINE, Operation',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00000',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'market for transport, passenger car, gasoline, type I',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 0.4,
                'type': 'technosphere',
                'code': '00001',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'market for transport, passenger car, gasoline, type II',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 0.6,
                'type': 'technosphere',
                'code': '00006',
                'unit': 'kilometer',
                'database': 'dummy_db'
            }
        ],
    },
    {
        'name': 'market for transport, passenger car, gasoline, type I',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00001',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'market for transport, passenger car, gasoline, type I',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00001',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'passenger car construction',
                'product': 'passenger car',
                'location': 'CH',
                'amount': 0,  # removed during double-counting correction
                'type': 'technosphere',
                'code': '00002',
                'unit': 'unit',
                'database': 'dummy_db'
            },
            {
                'name': 'market for petrol, low-sulfur',
                'product': 'petrol, low-sulfur',
                'location': 'CH',
                'amount': 0,  # removed during double-counting correction
                'type': 'technosphere',
                'code': '00003',
                'unit': 'kilogram',
                'database': 'dummy_db'
            },
            {
                'name': 'Carbon monoxide, fossil',
                'categories': ('air', 'non-urban air or from high stacks'),
                'amount': 0.05,
                'type': 'biosphere',
                'code': '00004',
                'unit': 'kilogram',
                'database': 'dummy_db_biosphere'
            },
            {
                'name': 'Carbon dioxide, fossil',
                'categories': ('air', 'non-urban air or from high stacks'),
                'amount': 0.2,
                'type': 'biosphere',
                'code': '00005',
                'unit': 'kilogram',
                'database': 'dummy_db_biosphere'
            },
            {
                'name': 'Occupation, unspecified',
                'categories': ('natural resource', 'land'),
                'amount': 10,
                'type': 'biosphere',
                'code': '00007',
                'unit': 'square meter-year',
                'database': 'dummy_db_biosphere'
            }
        ]
    },
    {
        'name': 'market for transport, passenger car, gasoline, type II',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00006',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'market for transport, passenger car, gasoline, type II',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00006',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'passenger car construction',
                'product': 'passenger car',
                'location': 'CH',
                'amount': 0,  # removed during double-counting correction
                'type': 'technosphere',
                'code': '00002',
                'unit': 'unit',
                'database': 'dummy_db'
            },
            {
                'name': 'market for petrol, low-sulfur',
                'product': 'petrol, low-sulfur',
                'location': 'CH',
                'amount': 0,  # removed during double-counting correction
                'type': 'technosphere',
                'code': '00003',
                'unit': 'kilogram',
                'database': 'dummy_db'
            },
            {
                'name': 'Carbon monoxide, fossil',
                'categories': ('air', 'non-urban air or from high stacks'),
                'amount': 0.04,
                'type': 'biosphere',
                'code': '00004',
                'unit': 'kilogram',
                'database': 'dummy_db_biosphere'
            },
            {
                'name': 'Carbon dioxide, fossil',
                'categories': ('air', 'non-urban air or from high stacks'),
                'amount': 0.15,
                'type': 'biosphere',
                'code': '00005',
                'unit': 'kilogram',
                'database': 'dummy_db_biosphere'
            },
            {
                'name': 'Occupation, unspecified',
                'categories': ('natural resource', 'land'),
                'amount': 15,
                'type': 'biosphere',
                'code': '00007',
                'unit': 'square meter-year',
                'database': 'dummy_db_biosphere'
            }
        ]
    },
    {
        'name': 'passenger car construction',
        'reference product': 'passenger car',
        'location': 'CH',
        'unit': 'unit',
        'database': 'dummy_db',
        'code': '00002',
        'classifications': [("CPC", "2345: Car production")],
    },
    {
        'name': 'market for petrol, low-sulfur',
        'reference product': 'petrol, low-sulfur',
        'location': 'CH',
        'unit': 'kilogram',
        'database': 'dummy_db',
        'code': '00003',
        'classifications': [("CPC", '10: Petroleum products')],
    },
]

model = [
    ['CAR_GASOLINE', 'MOB_PASSENGER', 1],
    ['CAR_GASOLINE', 'GASOLINE', -0.4],  # the "efficiency" in the ESM is 1/0.4 pkm/kWh
    ['CAR_GASOLINE', 'CO2_E', 0.2],
    ['CAR_GASOLINE', 'CONSTRUCTION', -1],
]

efficiency = [['CAR_GASOLINE', 'GASOLINE']]

mapping_esm_flows_to_CPC = [['GASOLINE', ['10: Petroleum products']]]

removed_flows = [
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', 0.05, 'kilogram', 'market for petrol, low-sulfur', 'petrol, low-sulfur', 'CH', 'dummy_db', '00003'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type II', 'CH',
     'dummy_db', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type II', 'CH',
     'dummy_db', 0.04, 'kilogram', 'market for petrol, low-sulfur', 'petrol, low-sulfur', 'CH', 'dummy_db', '00003'],
]  # the "efficiency" in the LCI database is 1/0.05 km/kg for type I and 1/0.04 km/kg for type II

unit_conversion = [
    ['CAR_GASOLINE', 'Operation', 0.667, 'kilometer', 'person kilometer'],
    ['petrol', 'Other', 0.0829, 'kilogram', 'kilowatt hour'],
]
# after unit conversion, the "efficiency" in the LCI db is 1/0.05 * 0.0829 / 0.667 pkm/kWh for type I
# and 1/0.04 * 0.0829 / 0.667 pkm/kWh for type II

double_counting_removal = [
    ['CAR_GASOLINE', 'GASOLINE', 0.05 * 0.4 + 0.04 * 0.6],
    ['CAR_GASOLINE', 'CONSTRUCTION', 5e-6 * 0.4 + 5e-6 * 0.6],
]

model = pd.DataFrame(model, columns=['Name', 'Flow', 'Amount'])

efficiency = pd.DataFrame(efficiency, columns=['Name', 'Flow'])

mapping_esm_flows_to_CPC = pd.DataFrame(mapping_esm_flows_to_CPC, columns=['Flow', 'CPC'])

removed_flows = pd.DataFrame(removed_flows, columns=[
    'Name', 'Product', 'Activity', 'Location', 'Database', 'Amount', 'Unit', 'Removed flow activity',
    'Removed flow product', 'Removed flow location', 'Removed flow database', 'Removed flow code'])

double_counting_removal = pd.DataFrame(double_counting_removal, columns=['Name', 'Flow', 'Amount'])

unit_conversion = pd.DataFrame(unit_conversion, columns=['Name', 'Type', 'Value', 'LCA', 'ESM'])


@pytest.mark.tags("workflow")
def test_correct_esm_and_lca_efficiency_differences():
    updated_db = correct_esm_and_lca_efficiency_differences(
        db=dummy_db,
        model=model,
        efficiency=efficiency,
        mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC,
        removed_flows=removed_flows,
        unit_conversion=unit_conversion,
        double_counting_removal=double_counting_removal,
    )

    updated_db_dict_name = database_list_to_dict(updated_db, 'name')
    efficiency_ratio = (1 / (0.05 * 0.4 + 0.04 * 0.6) * 0.0829 / 0.667) / (1 / 0.4)

    act_type_I = updated_db_dict_name[
        'market for transport, passenger car, gasoline, type I', 'transport, passenger car', 'CH', 'dummy_db']
    for bf in get_biosphere_flows(act_type_I):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert round(bf['amount'], 6) == round(0.05 * efficiency_ratio, 6)
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert round(bf['amount'], 6) == round(0.2 * efficiency_ratio, 6)
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 10

    act_type_II = updated_db_dict_name[
        'market for transport, passenger car, gasoline, type II', 'transport, passenger car', 'CH', 'dummy_db']
    for bf in get_biosphere_flows(act_type_II):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert round(bf['amount'], 6) == round(0.04 * efficiency_ratio, 6)
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert round(bf['amount'], 6) == round(0.15 * efficiency_ratio, 6)
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 15
