import pandas as pd
import pytest
from mescal.adapt_efficiency import correct_esm_and_lca_efficiency_differences
from mescal.utils import database_list_to_dict, get_biosphere_flows

# LCI database after double-counting correction
dummy_db = [
    {
        'name': 'CAR_BIODIESEL_B20, Operation',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00000B',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'CAR_BIODIESEL_B20, Operation',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00000B',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'market for transport, passenger car, diesel',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'technosphere',
                'code': '00001B',
                'unit': 'kilometer',
                'database': 'dummy_db'
            }
        ],
    },
    {
        'name': 'CAR_PROPANE, Operation',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00000A',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'CAR_PROPANE, Operation',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00000A',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
            {
                'name': 'market for transport, passenger car, gasoline, type I',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'technosphere',
                'code': '00001A',
                'unit': 'kilometer',
                'database': 'dummy_db'
            },
        ],
    },
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
        'name': 'market for transport, passenger car, gasoline, type I',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00001A',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'market for transport, passenger car, gasoline, type I',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00001A',
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
        'name': 'market for transport, passenger car, diesel',
        'reference product': 'transport, passenger car',
        'unit': 'kilometer',
        'database': 'dummy_db',
        'location': 'CH',
        'code': '00001B',
        'classifications': [("CPC", "1234: Car transportation")],
        'exchanges': [
            {
                'name': 'market for transport, passenger car, diesel',
                'product': 'transport, passenger car',
                'location': 'CH',
                'amount': 1,
                'type': 'product',
                'code': '00001B',
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
                'name': 'market for diesel',
                'product': 'petrol, low-sulfur',
                'location': 'CH',
                'amount': 0,  # removed during double-counting correction
                'type': 'technosphere',
                'code': '00010',
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
    {
        'name': 'market for diesel',
        'reference product': 'diesel',
        'location': 'CH',
        'unit': 'kilogram',
        'database': 'dummy_db',
        'code': '00010',
        'classifications': [("CPC", '10: Petroleum products')],
    }
]

model = [
    ['CAR_GASOLINE', 'MOB_PASSENGER', 1],
    ['CAR_GASOLINE', 'GASOLINE', -0.4],  # the "efficiency" in the ESM is 1/0.4 pkm/kWh
    ['CAR_GASOLINE', 'CO2_E', 0.2],
    ['CAR_GASOLINE', 'CONSTRUCTION', -1],
    ['CAR_PROPANE', 'MOB_PASSENGER', 1],
    ['CAR_PROPANE', 'PROPANE', -0.8],  # the "efficiency" in the ESM is 1/0.8 pkm/kWh
    ['CAR_PROPANE', 'CO2_E', 0.2],
    ['CAR_PROPANE', 'CONSTRUCTION', -1],
    ['CAR_BIODIESEL_B20', 'MOB_PASSENGER', 1],
    ['CAR_BIODIESEL_B20', 'BIO_DIESEL', -0.1],
    ['CAR_BIODIESEL_B20', 'DIESEL', -0.4],  # the "efficiency" in the ESM is 1/(0.1+0.4) pkm/kWh
    ['CAR_BIODIESEL_B20', 'CO2_E', 0.2],
    ['CAR_BIODIESEL_B20', 'CONSTRUCTION', -1],
]

efficiency = [
    ['CAR_GASOLINE', "['GASOLINE']"],
    ['CAR_PROPANE', "['PROPANE']"],
    ['CAR_BIODIESEL_B20', "['BIO_DIESEL', 'DIESEL']"],
]

mapping_esm_flows_to_CPC = [
    ['GASOLINE', ['10: Petroleum products']],
    ['PROPANE', ['15: Propane']],
    ['BIO_DIESEL', ['20: Biodiesel']],
    ['DIESEL', ['10: Petroleum products']],
]

removed_flows = [
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', '00001', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', '00001', 0.05, 'kilogram', 'market for petrol, low-sulfur', 'petrol, low-sulfur', 'CH', 'dummy_db',
     '00003'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type II', 'CH',
     'dummy_db', '00006', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_GASOLINE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type II', 'CH',
     'dummy_db', '00006', 0.04, 'kilogram', 'market for petrol, low-sulfur', 'petrol, low-sulfur', 'CH', 'dummy_db',
     '00003'],
    ['CAR_PROPANE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', '00001A', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_PROPANE', 'transport, passenger car', 'market for transport, passenger car, gasoline, type I', 'CH',
     'dummy_db', '00001A', 0.05, 'kilogram', 'market for petrol, low-sulfur', 'petrol, low-sulfur', 'CH', 'dummy_db',
     '00003'],
    ['CAR_BIODIESEL_B20', 'transport, passenger car', 'market for transport, passenger car, diesel', 'CH', 'dummy_db',
     '00001B', 5e-6, 'unit', 'passenger car construction', 'passenger car', 'CH', 'dummy_db', '00002'],
    ['CAR_BIODIESEL_B20', 'transport, passenger car', 'market for transport, passenger car, diesel', 'CH', 'dummy_db',
     '00001B', 0.06, 'kilogram', 'market for diesel', 'diesel', 'CH', 'dummy_db', '00010'],
]
# CAR_GASOLINE: the "efficiency" in the LCI database is 1/0.05 km/kg for type I and 1/0.04 km/kg for type II
# CAR_PROPANE: the right flow has not been found (gasoline instead of propane), so the efficiency should not be adjusted
# CAR_BIODIESEL_B20: the "efficiency" in the LCI database is 1/0.06 km/kg

unit_conversion = [
    ['CAR_GASOLINE', 'Operation', 0.667, 'kilometer', 'person kilometer'],
    ['CAR_PROPANE', 'Operation', 0.667, 'kilometer', 'person kilometer'],
    ['CAR_BIODIESEL_B20', 'Operation', 0.667, 'kilometer', 'person kilometer'],
    ['petrol', 'Other', 0.0829, 'kilogram', 'kilowatt hour'],
    ['diesel', 'Other', 0.075, 'kilogram', 'kilowatt hour'],
]
# CAR_GASOLINE: after unit conversion, the "efficiency" in the LCI db is 1/0.05 * 0.0829/0.667 pkm/kWh for type I
# and 1/0.04 * 0.0829 / 0.667 pkm/kWh for type II
# CAR_BIODIESEL_B20: after unit conversion, the "efficiency" in the LCI db is 1/0.06 * 0.075/0.667 pkm/kWh

double_counting_removal = [
    ['CAR_GASOLINE', 'GASOLINE', 0.05 * 0.4 + 0.04 * 0.6],
    ['CAR_GASOLINE', 'TRANSPORT_FUEL', 0.05 * 0.4 + 0.04 * 0.6],
    ['CAR_GASOLINE', 'CONSTRUCTION', 5e-6 * 0.4 + 5e-6 * 0.6],
    ['CAR_PROPANE', 'TRANSPORT_FUEL', 0.05],
    ['CAR_PROPANE', 'CONSTRUCTION', 5e-6],
    ['CAR_BIODIESEL_B20', 'DIESEL', 0.06],
    ['CAR_BIODIESEL_B20', 'TRANSPORT_FUEL', 0.06],
    ['CAR_BIODIESEL_B20', 'CONSTRUCTION', 5e-6],
]

model = pd.DataFrame(model, columns=['Name', 'Flow', 'Amount'])

efficiency = pd.DataFrame(efficiency, columns=['Name', 'Flow'])

mapping_esm_flows_to_CPC = pd.DataFrame(mapping_esm_flows_to_CPC, columns=['Flow', 'CPC'])

removed_flows = pd.DataFrame(removed_flows, columns=[
    'Name', 'Product', 'Activity', 'Location', 'Database', 'Code', 'Amount', 'Unit', 'Removed flow activity',
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

    updated_db_dict_code = database_list_to_dict(updated_db, 'code')
    efficiency_ratio = (1 / (0.05 * 0.4 + 0.04 * 0.6) * 0.0829 / 0.667) / (1 / 0.4)

    act_type_I = updated_db_dict_code['dummy_db', '00001']
    for bf in get_biosphere_flows(act_type_I):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert round(bf['amount'], 6) == round(0.05 * efficiency_ratio, 6)
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert round(bf['amount'], 6) == round(0.2 * efficiency_ratio, 6)
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 10

    act_type_II = updated_db_dict_code['dummy_db', '00006']
    for bf in get_biosphere_flows(act_type_II):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert round(bf['amount'], 6) == round(0.04 * efficiency_ratio, 6)
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert round(bf['amount'], 6) == round(0.15 * efficiency_ratio, 6)
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 15

    act = updated_db_dict_code['dummy_db', '00001A']
    for bf in get_biosphere_flows(act):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert bf['amount'] == 0.05
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert bf['amount'] == 0.2
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 10

    act = updated_db_dict_code['dummy_db', '00001B']
    efficiency_ratio = (1 / 0.06 * 0.075 / 0.667) / (1 / 0.5)
    for bf in get_biosphere_flows(act):
        if bf['name'] == 'Carbon monoxide, fossil':
            assert round(bf['amount'], 6) == round(0.05 * efficiency_ratio, 6)
        elif bf['name'] == 'Carbon dioxide, fossil':
            assert round(bf['amount'], 6) == round(0.2 * efficiency_ratio, 6)
        elif bf['name'] == 'Occupation, unspecified':
            assert bf['amount'] == 10
