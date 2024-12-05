import pandas as pd
from .normalization import restrict_lcia_metrics
from pathlib import Path


@staticmethod
def generate_mod_file_ampl(
        impact_abbrev: pd.DataFrame,
        lcia_methods: list[str],
        specific_lcia_methods: list[str] = None,
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
        path: str = 'results/',
        file_name: str = 'objectives',
        metadata: dict = None
) -> None:
    """
    Create an AMPL mod file containing everything related to LCA

    :param lcia_methods: LCIA methods to be used
    :param specific_lcia_methods: specific LCIA methods to be used
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :param impact_abbrev: dataframe containing the impact abbreviations of the LCIA method
    :param path: path where the mod file will be saved
    :param file_name: name of the .mod file
    :param metadata: dictionary containing the metadata to be written at the beginning of the file
    :return: None (writes the file)
    """

    if metadata is None:
        metadata = {}

    impact_abbrev = restrict_lcia_metrics(
        df=impact_abbrev,
        lcia_methods=lcia_methods,
        specific_lcia_categories=specific_lcia_categories,
        specific_lcia_abbrev=specific_lcia_abbrev,
    )

    Path(path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist

    with open(f'{path}{file_name}.mod', 'w') as f:

        # Write metadata at the beginning of the file
        if 'lcia_method' in metadata:
            f.write(f'# LCIA method: {metadata["lcia_method"]}\n')
        f.write('\n')

        # Set of LCA indicators
        f.write('set INDICATORS;\n\n')

        # Declaring the LCIA parameters and variables
        f.write('param lcia_constr {INDICATORS,TECHNOLOGIES} default 1e-12;\n'
                'param lcia_op {INDICATORS,TECHNOLOGIES} default 1e-12;\n'
                'param lcia_res {INDICATORS, RESOURCES} default 1e-12;\n'
                'param refactor {INDICATORS} default 1;\n'
                'var LCIA_constr {INDICATORS,TECHNOLOGIES};\n'
                'var LCIA_op {INDICATORS,TECHNOLOGIES};\n'
                'var LCIA_res {INDICATORS,RESOURCES};\n'
                'var TotalLCIA {INDICATORS} >= 0;\n\n')

        # Equation of LCIAs variables (construction scaling to F_Mult)
        f.write('# LCIA construction\n'
                'subject to lcia_constr_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                f'  LCIA_constr[id,i] >= (1/refactor[id]) * lcia_constr[id,i] * F_Mult[i];\n\n')

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
