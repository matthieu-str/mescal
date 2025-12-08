import pandas as pd
import pytest
from mescal import *

dummy_main_db = [
    {
        'name': 'dummy process 1',
        'location': 'CA-QC',
        'code': '10000',
        'database': 'dummy_main_db',
    },
    {
        'name': 'dummy process 2',
        'location': 'CH',
        'code': '10001',
        'database': 'dummy_main_db',
    },
    {
        'name': 'dummy process 3',
        'location': 'CA-QC',
        'code': '10002',
        'database': 'dummy_main_db',
    },
    {
        'name': 'dummy process 4',
        'location': 'FR',
        'code': '10003',
        'database': 'dummy_main_db',
    },
]

dummy_esm_db = [
    {
        'name': 'BUS_DIESEL, Operation',
        'location': 'CAN',
        'code': '00000',
        'database': 'dummy_esm_db',
    },
    {
        'name': 'BUS_DIESEL, Construction',
        'location': 'CAN',
        'code': '00001',
        'database': 'dummy_esm_db',
    },
    {
        'name': 'BUS_EV, Operation',
        'location': 'CAN',
        'code': '00002',
        'database': 'dummy_esm_db',
    },
    {
        'name': 'BUS_EV, Construction',
        'location': 'CAN',
        'code': '00003',
        'database': 'dummy_esm_db',
    },
]

R = [
    [('LCIA', 'method 1'), '00000', 1000, 'BUS_DIESEL', 'Operation'],
    [('LCIA', 'method 2'), '00000', 500, 'BUS_DIESEL', 'Operation'],
    [('LCIA', 'method 1'), '00001', 100, 'BUS_DIESEL', 'Construction'],
    [('LCIA', 'method 2'), '00001', 1200, 'BUS_DIESEL', 'Construction'],
    [('LCIA', 'method 1'), '00002', 200, 'BUS_EV', 'Operation'],
    [('LCIA', 'method 2'), '00002', 250, 'BUS_EV', 'Operation'],
    [('LCIA', 'method 1'), '00003', 300, 'BUS_EV', 'Construction'],
    [('LCIA', 'method 2'), '00003', 2000, 'BUS_EV', 'Construction'],
]

impact_abbrev = [
    [('LCIA', 'method 1'), 'unit 1', 'M1'],
    [('LCIA', 'method 2'), 'unit 2', 'M2'],
]

contrib_processes = [
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 300, 1, '00000', 'dummy_esm_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 220, 5, '10001', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 160, 5, '10002', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 140, 5, '10003', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 55, 10, '10002', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 50, 1, '10000', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 40, 2, '10001', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 15, 3, '10003', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 150, 1, '00000', 'dummy_esm_db'],
    ['processes', ('LCIA', 'method 2'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 110, 3, '10001', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 75, 4, '10002', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Operation', 55, 3, '10000', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 70, 3, '10001', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 55, 4, '10002', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 30, 2, '10000', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 2'), '00002', 'dummy_esm_db', 'BUS_EV', 'Operation', 15, 15, '10003', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Construction', 30, 3, '10000', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Construction', 25, 0.5, '10001', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Construction', 15, 1, '10002', 'dummy_main_db'],
    ['processes', ('LCIA', 'method 1'), '00000', 'dummy_esm_db', 'BUS_DIESEL', 'Construction', 5, 0.1, '10003', 'dummy_main_db'],
]

R = pd.DataFrame(R, columns=['Impact_category', 'New_code', 'Value', 'Name', 'Type'])
impact_abbrev = pd.DataFrame(impact_abbrev, columns=['Impact_category', 'Unit', 'Abbrev'])
contrib_processes = pd.DataFrame(contrib_processes, columns=[
    'contribution_type', 'impact_category', 'act_code', 'act_database', 'act_name', 'act_type', 'score', 'amount',
    'code', 'database',
])


@pytest.mark.tags("workflow")
def test_normalize_lca_metrics():

    esm = ESM(
        mapping=pd.DataFrame(),
        technology_compositions=pd.DataFrame(),
        unit_conversion=pd.DataFrame(),
        lifetime=pd.DataFrame(),
        main_database=Database(db_as_list=dummy_main_db),
        main_database_name='dummy_main_db',
        esm_db_name='dummy_esm_db',
        model=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
        esm_location='CA-QC',
    )

    esm.esm_db = Database(db_as_list=dummy_esm_db)

    R_normalized = esm.normalize_lca_metrics(
        R=R,
        mip_gap=1e-6,
        lcia_methods=['LCIA'],
        impact_abbrev=impact_abbrev,
        output='return',
    )

    # Method 1
    bus_diesel_op_1_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'M1')
    ].Value_norm.values[0]

    bus_diesel_constr_1_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'M1')
    ].Value_norm.values[0]

    bus_ev_op_1_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'M1')
    ].Value_norm.values[0]

    bus_ev_constr_1_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'M1')
    ].Value_norm.values[0]

    # Method 2
    bus_diesel_op_2_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'M2')
    ].Value_norm.values[0]

    bus_diesel_constr_2_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'M2')
    ].Value_norm.values[0]

    bus_ev_op_2_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'M2')
    ].Value_norm.values[0]

    bus_ev_constr_2_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'M2')
    ].Value_norm.values[0]

    assert bus_diesel_op_1_normalized == 1000 / 1000
    assert bus_diesel_constr_1_normalized == 100 / 1000
    assert bus_ev_op_1_normalized == 200 / 1000
    assert bus_ev_constr_1_normalized == 300 / 1000

    assert bus_diesel_op_2_normalized == 500 / 500
    assert bus_diesel_constr_2_normalized == 1200 / 500
    assert bus_ev_op_2_normalized == 250 / 500
    assert bus_ev_constr_2_normalized == 2000 / 500

    R_territorial = esm.normalize_lca_metrics(
        R=R,
        assessment_type='territorial emissions',
        contrib_processes=contrib_processes,
        mip_gap=1e-6,
        lcia_methods=['LCIA'],
        impact_abbrev=impact_abbrev,
        output='return',
    )

    bus_diesel_op_1_terr = R_territorial[
        (R_territorial.Name == 'BUS_DIESEL')
        & (R_territorial.Type == 'Operation')
        & (R_territorial.Abbrev == 'M1')
    ].Value_norm.values[0]

    bus_ev_op_1_terr = R_territorial[
        (R_territorial.Name == 'BUS_EV')
        & (R_territorial.Type == 'Operation')
        & (R_territorial.Abbrev == 'M1')
    ].Value_norm.values[0]

    bus_diesel_op_2_terr = R_territorial[
        (R_territorial.Name == 'BUS_DIESEL')
        & (R_territorial.Type == 'Operation')
        & (R_territorial.Abbrev == 'M2')
    ].Value_norm.values[0]

    bus_ev_op_2_terr = R_territorial[
        (R_territorial.Name == 'BUS_EV')
        & (R_territorial.Type == 'Operation')
        & (R_territorial.Abbrev == 'M2')
    ].Value_norm.values[0]

    bus_diesel_constr_1_terr = R_territorial[
        (R_territorial.Name == 'BUS_DIESEL')
        & (R_territorial.Type == 'Construction')
        & (R_territorial.Abbrev == 'M1')
    ].Value_norm.values[0]

    assert bus_diesel_op_1_terr == (300 + 160) / 1000
    assert bus_ev_op_1_terr == (55 + 50) / 1000
    assert bus_diesel_op_2_terr == (150 + 75 + 55) / 500
    assert bus_ev_op_2_terr == (55 + 30) / 500
    assert bus_diesel_constr_1_terr == (30 + 15) / 1000