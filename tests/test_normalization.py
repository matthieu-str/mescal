import pandas as pd
import pytest
from mescal import *

R = [
    [('LCIA', 'method 1'), '00000', 1000, 'BUS_DIESEL', 'Operation'],
    [('LCIA', 'method 2'), '00000', 500, 'BUS_DIESEL', 'Operation'],
    [('LCIA', 'method 1'), '00000', 100, 'BUS_DIESEL', 'Construction'],
    [('LCIA', 'method 2'), '00000', 1200, 'BUS_DIESEL', 'Construction'],
    [('LCIA', 'method 1'), '00001', 200, 'BUS_EV', 'Operation'],
    [('LCIA', 'method 2'), '00001', 250, 'BUS_EV', 'Operation'],
    [('LCIA', 'method 1'), '00001', 300, 'BUS_EV', 'Construction'],
    [('LCIA', 'method 2'), '00001', 2000, 'BUS_EV', 'Construction'],
]
impact_abbrev = [
    [('LCIA', 'method 1'), 'unit 1', 'M1'],
    [('LCIA', 'method 2'), 'unit 2', 'M2'],
]

R = pd.DataFrame(R, columns=['Impact_category', 'New_code', 'Value', 'Name', 'Type'])
impact_abbrev = pd.DataFrame(impact_abbrev, columns=['Impact_category', 'Unit', 'Abbrev'])


@pytest.mark.tags("workflow")
def test_normalize_lca_metrics():

    esm = ESM(
        mapping=pd.DataFrame(),
        technology_compositions=pd.DataFrame(),
        unit_conversion=pd.DataFrame(),
        lifetime=pd.DataFrame(),
        main_database=Database(db_as_list=[]),
        main_database_name='',
        esm_db_name='dummy_esm_db',
        model=pd.DataFrame(),
        mapping_esm_flows_to_CPC_cat=pd.DataFrame(),
    )

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