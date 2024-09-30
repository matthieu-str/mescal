import bw2data as bd
import pytest
from mescal.modify_inventory import change_dac_biogenic_carbon_flow, change_fossil_carbon_flows_of_biofuels
from mescal.database import Database

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
    },
    {
        "name": "CAR_BIODIESEL, Operation",
        "reference product": "transport, passenger car",
        "location": "GLO",
        "database": "dummy_esm_db_dac",
        "unit": "kilometer",
        "code": "00002",
        "exchanges": [
            {
                "name": "CAR_BIODIESEL, Operation",
                "product": "transport, passenger car",
                "location": "GLO",
                "database": "dummy_esm_db_dac",
                "type": "production",
                "amount": 1,
                "unit": "km",
                "code": "00002"
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


@pytest.mark.tags("requires_ecoinvent")
def test_change_carbon_flow():
    bd.projects.set_current('ecoinvent3.9.1')
    Database(db_as_list=dummy_esm_db).write_to_brightway("dummy_esm_db_dac")

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

    change_fossil_carbon_flows_of_biofuels(db_name="dummy_esm_db_dac",
                                           activity_name="CAR_BIODIESEL, Operation",
                                           activity_code="00002",
                                           biogenic_ratio=0.80)

    act_car = [i for i in bd.Database("dummy_esm_db_dac").search("CAR_BIODIESEL, Operation", limit=1000) if (
        ("CAR_BIODIESEL, Operation" == i.as_dict()['name'])
    )][0]

    biosphere_flows_car = [i for i in act_car.biosphere()]

    assert len(biosphere_flows_car) == 6

    carbon_dio_fossil_air = [i for i in biosphere_flows_car if
                             (i.as_dict()['name'] == 'Carbon dioxide, fossil')
                             & (i.as_dict()['categories'] == ('air',))][0]
    carbon_dio_non_fossil_air = [i for i in biosphere_flows_car if
                                 (i.as_dict()['name'] == 'Carbon dioxide, non-fossil')
                                 & (i.as_dict()['categories'] == ('air',))][0]
    carbon_dio_fossil_urban = [i for i in biosphere_flows_car if
                               (i.as_dict()['name'] == 'Carbon dioxide, fossil')
                               & (i.as_dict()['categories'] == ('air', 'urban air close to ground'))][0]
    carbon_dio_non_fossil_urban = [i for i in biosphere_flows_car if
                                   (i.as_dict()['name'] == 'Carbon dioxide, non-fossil')
                                   & (i.as_dict()['categories'] == ('air', 'urban air close to ground'))][0]
    carbon_mono_fossil = [i for i in biosphere_flows_car if i.as_dict()['name'] == 'Carbon monoxide, fossil'][0]
    carbon_mono_non_fossil = [i for i in biosphere_flows_car if i.as_dict()['name'] == 'Carbon monoxide, non-fossil'][0]

    assert carbon_dio_fossil_air.as_dict()['amount'] == 0.20 * (1 - 0.80)
    assert carbon_dio_non_fossil_air.as_dict()['amount'] == 0.20 * 0.80
    assert carbon_dio_fossil_urban.as_dict()['amount'] == 0.30 * (1 - 0.80)
    assert carbon_dio_non_fossil_urban.as_dict()['amount'] == 0.30 * 0.80
    assert carbon_mono_fossil.as_dict()['amount'] == 0.10 * (1 - 0.80)
    assert carbon_mono_non_fossil.as_dict()['amount'] == 0.10 * 0.80

    assert carbon_dio_fossil_air.as_dict()['input'] == ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
    assert carbon_dio_non_fossil_air.as_dict()['input'] == ('biosphere3', 'eba59fd6-f37e-41dc-9ca3-c7ea22d602c7')
    assert carbon_dio_fossil_urban.as_dict()['input'] == ('biosphere3', 'f9749677-9c9f-4678-ab55-c607dfdc2cb9')
    assert carbon_dio_non_fossil_urban.as_dict()['input'] == ('biosphere3', '73ed05cc-9727-4abf-9516-4b5c0fe54a16')
    assert carbon_mono_fossil.as_dict()['input'] == ('biosphere3', 'ba2f3f82-c93a-47a5-822a-37ec97495275')
    assert carbon_mono_non_fossil.as_dict()['input'] == ('biosphere3', '2cb2333c-1599-46cf-8435-3dffce627524')

