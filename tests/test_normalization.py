import pandas as pd
from mescal.normalization import normalize_lca_metrics
import pytest

R = [
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Ecosystem quality', 'Total ecosystem quality'), '00000', 1000,
     'BUS_DIESEL', 'Operation'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Human health', 'Total human health'), '00000', 500, 'BUS_DIESEL',
     'Operation'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Ecosystem quality', 'Total ecosystem quality'), '00000', 100,
     'BUS_DIESEL', 'Construction'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Human health', 'Total human health'), '00000', 10, 'BUS_DIESEL',
     'Construction'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Ecosystem quality', 'Total ecosystem quality'), '00001', 200,
     'BUS_EV', 'Operation'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Human health', 'Total human health'), '00001', 250, 'BUS_EV',
     'Operation'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Ecosystem quality', 'Total ecosystem quality'), '00001', 300,
     'BUS_EV', 'Construction'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Human health', 'Total human health'), '00001', 20, 'BUS_EV',
     'Construction'],
]
impact_abbrev = [
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Ecosystem quality', 'Total ecosystem quality'), 'PDF.m2.yr', 'TTEQ', 'EQ'],
    [('IMPACT World+ Damage 2.0.1_regionalized', 'Human health', 'Total human health'), 'DALY', 'TTHH', 'HH'],
]

R = pd.DataFrame(R, columns=['Impact_category', 'New_code', 'Value', 'Name', 'Type'])
impact_abbrev = pd.DataFrame(impact_abbrev, columns=['Impact_category', 'Unit', 'Abbrev', 'AoP'])


@pytest.mark.tags("workflow")
def test_normalize_lca_metrics():
    R_normalized, refactor = normalize_lca_metrics(
        R=R,
        mip_gap=1e-6,
        lcia_methods=['IMPACT World+ Damage 2.0.1_regionalized'],
        specific_lcia_abbrev=['TTHH', 'TTEQ'],
        impact_abbrev=impact_abbrev,
        output='return',
    )
    # refactor should be the highest operation metric over the highest construction metric
    assert refactor['HH'] == 500/20
    assert refactor['EQ'] == 1000/300

    bus_diesel_op_tthh_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'TTHH')
        ].Value_norm.values[0]

    bus_diesel_constr_tthh_normalized = R_normalized[
        (R_normalized.Name == 'BUS_DIESEL')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'TTHH')
        ].Value_norm.values[0]

    bus_ev_op_tthh_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Operation')
        & (R_normalized.Abbrev == 'TTHH')
        ].Value_norm.values[0]

    bus_ev_constr_tthh_normalized = R_normalized[
        (R_normalized.Name == 'BUS_EV')
        & (R_normalized.Type == 'Construction')
        & (R_normalized.Abbrev == 'TTHH')
        ].Value_norm.values[0]

    assert bus_diesel_op_tthh_normalized == 1
    assert bus_ev_constr_tthh_normalized == 1
    assert bus_ev_op_tthh_normalized == 250 / 500
    assert bus_diesel_constr_tthh_normalized == 10 * (500/20) / 500
