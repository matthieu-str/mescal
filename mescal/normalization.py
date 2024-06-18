import pandas as pd
import ast


def tech_type(tech: str) -> str:
    """
    Returns the short name of the technology type
    :param tech: (str) technology type
    :return: (str) short name of the technology type
    """
    if tech == 'Construction':
        return 'constr'
    elif tech == 'Operation':
        return 'op'
    elif tech == 'Resource':
        return 'res'
    else:
        raise ValueError(f"Unknown technology type: {tech}")


def lcia_methods_short_names(lcia_method: str) -> str:
    """
    Returns the short name of the LCIA method
    :param lcia_method: (str) LCIA method
    :return: (str) short name of the LCIA method
    """
    if lcia_method == 'IMPACT World+ Damage 2.0.1':
        return 'endpoint'
    elif lcia_method == 'IMPACT World+ Damage 2.0.1 - Total only':
        return 'endpoint_tot'
    elif lcia_method == 'IMPACT World+ Midpoint 2.0.1':
        return 'midpoint'
    elif lcia_method == 'IMPACT World+ Footprint 2.0.1':
        return 'footprint'
    else:
        raise ValueError(f"Unknown LCIA method: {lcia_method}")


def normalize_lca_metrics(R: pd.DataFrame, f_norm: float, mip_gap: float, refactor: float, lcia_method: str,
                          impact_abbrev: pd.DataFrame, biogenic: bool = False, path: str = 'results/') -> None:
    """
    Create a .dat file containing the normalized LCA metrics for AMPL and a csv file containing the normalization
    factors
    :param path: (str) path to results folder
    :param R: (pd.DataFrame) dataframe containing the LCA results
    :param f_norm: (float) maximum allowed order of magnitude between the smallest and the largest value LCA metrics
    within an AoP
    :param mip_gap: (float) values outside the f_norm threshold will be set to mip_gap / f_norm
    :param refactor: (float) value of refactor to apply for construction metrics for better convergence
    :param lcia_method: (str) LCIA method to be used
    :param impact_abbrev: (pd.DataFrame) dataframe containing the impact categories abbreviations
    :param biogenic: (boolean) whether biogenic carbon flows impact assessment method should be included or not
    :return: None
    """
    R_constr = R[R['Type'] == 'Construction']
    R_constr['Value'] *= refactor

    R_op_res = R[R['Type'] != 'Construction']

    R = pd.concat([R_constr, R_op_res])

    if not biogenic:
        list_biogenic_cat = ["CFB", "REQDB", "m_CCLB", "m_CCSB", "TTEQB", "TTHHB", "CCEQSB", "CCEQLB", "CCHHSB",
                             "CCHHLB", "MALB", "MASB"]
        impact_abbrev.drop(impact_abbrev[impact_abbrev.Abbrev.isin(list_biogenic_cat)].index, inplace=True)

    R.Impact_category = R.Impact_category.apply(lambda x: ast.literal_eval(x))
    impact_abbrev.Impact_category = impact_abbrev.Impact_category.apply(lambda x: ast.literal_eval(x))

    if lcia_method == 'IMPACT World+ Damage 2.0.1 - Total only':
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x:
                                                          x.Impact_category[0] == 'IMPACT World+ Damage 2.0.1', axis=1)]
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x: 'Total' in x.Impact_category[2], axis=1)]
    else:
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x: x.Impact_category[0] == lcia_method, axis=1)]

    R = pd.merge(R, impact_abbrev, on='Impact_category')
    R['max_AoP'] = R.groupby('AoP')['Value'].transform('max')
    R['Value_norm'] = R['Value'] / R['max_AoP']
    R['Value_norm'] = R['Value_norm'].apply(lambda x: x if x > mip_gap else mip_gap / f_norm)

    with open(f'{path}techs_lcia_{lcia_methods_short_names(lcia_method)}.dat', 'w') as f:
        f.write(f"set INDICATORS := {' '.join(R['Abbrev'].unique())};\n")
        for i in range(len(R)):
            f.write(f"let lcia_{tech_type(R.Type.iloc[i])}['{R.Abbrev.iloc[i]}','{R.Name.iloc[i]}'] := "
                    f"{R.Value_norm.iloc[i]}; #{R.Unit.iloc[i]} (normalized)\n")

    R[['AoP', 'max_AoP']].drop_duplicates().to_csv(f'{path}res_lcia_max_{lcia_methods_short_names(lcia_method)}.csv',
                                                   index=False)  # to come back to the original values
