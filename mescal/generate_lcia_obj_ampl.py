import pandas as pd
from .normalization import lcia_methods_short_names


def gen_lcia_obj(path: str, lcia_method: str, refactor: str, impact_abbrev: pd.DataFrame, biogenic: bool = False) \
        -> None:
    """
    Create an AMPL mod file containing everything related to LCA
    :param path: (str) path to EnergyScope AMPL folder
    :param lcia_method: (str) lcia method to be used, can be 'midpoints', 'endpoints' or 'endpoints_tot'
    :param refactor: (str) value of refactor to apply for construction metrics
    :param impact_abbrev: (pd.DataFrame) dataframe containing the impact abbreviations
    :param biogenic: (boolean) whether biogenic carbon flows impact assessment method should be included or not
    :return: None
    """

    if not biogenic:
        list_biogenic_cat = ["CFB", "REQDB", "m_CCLB", "m_CCSB", "TTEQB", "TTHHB", "CCEQSB", "CCEQLB", "CCHHSB",
                             "CCHHLB", "MALB", "MASB"]
        impact_abbrev.drop(impact_abbrev[impact_abbrev.Abbrev.isin(list_biogenic_cat)].index, inplace=True)

    if lcia_method == 'IMPACT World+ Damage 2.0.1 - Total only':
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x:
                                                          x.Impact_category[0] == 'IMPACT World+ Damage 2.0.1', axis=1)]
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x: 'Total' in x.Impact_category[2], axis=1)]
    else:
        impact_abbrev = impact_abbrev[impact_abbrev.apply(lambda x: x.Impact_category[0] == lcia_method, axis=1)]

    with open(f'{path}objectives_{lcia_methods_short_names(lcia_method)}.mod', 'w') as f:

        # Set of LCA indicators
        f.write('set INDICATORS;\n\n')

        # Declaring the LCIA parameters and variables
        f.write('param lcia_constr {INDICATORS,TECHNOLOGIES} default 1e-12;\n'
                'param lcia_op {INDICATORS,TECHNOLOGIES} default 1e-12;\n'
                'param lcia_res {INDICATORS, RESOURCES} default 1e-12;\n'
                'var LCIA_constr{INDICATORS,TECHNOLOGIES};\n'
                'var LCIA_op{INDICATORS,TECHNOLOGIES};\n'
                'var LCIA_res{INDICATORS,RESOURCES};\n'
                'var TotalLCIA{INDICATORS} >= 0;\n\n')

        # Equation of LCIAs variables (construction scaling to F_Mult)
        f.write('# LCIA construction\n'
                'subject to lcia_constr_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                f'  LCIA_constr[id,i] >= (1/{refactor}) * lcia_constr[id,i] * F_Mult[i];\n\n')

        # Equation of LCIAs variables (operation scaling to F_Mult_t)
        f.write('# LCIA operation\n'
                'subject to lcia_op_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                '  LCIA_op[id,i] >= lcia_op[id,i] * sum {t in PERIODS} (t_op[t] * F_Mult_t[i, t]);\n\n')

        # Equation of LCIAs variables (resources scaling to F_Mult_t)
        f.write('# LCIA resources\n'
                'subject to lcia_res_calc {id in INDICATORS, r in RESOURCES}:\n'
                '  LCIA_res[id,r] >= lcia_res[id,r] * sum {t in PERIODS} (t_op[t] * F_Mult_t[r, t]);\n\n')

        # Equation defining the total LCIA impact (sum over all technologies and resources)
        f.write('subject to totalLCIA_calc_r {id in INDICATORS}:\n'
                '  TotalLCIA[id] = sum {i in TECHNOLOGIES} (LCIA_constr[id,i] / lifetime[i]  '
                '+ LCIA_op[id,i]) + sum{r in RESOURCES} (LCIA_res[id,r]);\n\n')

        # Declaring the total LCIA amount variables
        for abbrev in list(impact_abbrev.Abbrev):
            f.write(f'var TotalLCIA_{abbrev};\n'
                    f'subject to LCIA_{abbrev}_cal:\n'
                    f"  TotalLCIA_{abbrev} = TotalLCIA['{abbrev}'] + TotalCost*1e-6;\n\n")
