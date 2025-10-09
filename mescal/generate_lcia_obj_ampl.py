import pandas as pd
from .normalization import restrict_lcia_metrics, from_str_to_tuple
from pathlib import Path


def generate_mod_file_ampl(
        self,
        impact_abbrev: pd.DataFrame,
        lcia_methods: list[str],
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
        assessment_type: str = 'esm',
        path: str = 'results/',
        file_name: str = 'objectives',
        metadata: dict = None
) -> None:
    """
    Create an AMPL mod file containing everything related to LCA

    :param lcia_methods: LCIA methods to be used
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :param assessment_type: type of assessment, can be 'esm' for the full LCA database, or 'direct emissions' for the
        computation of territorial emissions only
    :param impact_abbrev: dataframe containing the impact abbreviations of the LCIA method
    :param path: path where the mod file will be saved
    :param file_name: name of the .mod file
    :param metadata: dictionary containing the metadata to be written at the beginning of the file
    :return: None (writes the file)
    """

    if assessment_type == 'esm':
        metric_type = 'LCIA'
    elif assessment_type == 'direct emissions':
        metric_type = 'DIRECT'
    else:
        raise ValueError(f"Unknown assessment type: {assessment_type}. Must be 'esm' or 'direct emissions'.")

    if metadata is None:
        metadata = {}

    impact_abbrev = from_str_to_tuple(impact_abbrev, 'Impact_category')

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

        if assessment_type == 'esm':

            # Set of LCA indicators
            f.write('set INDICATORS;\n\n')

        if self.pathway:
            # Declaring the LCIA parameters and variables
            if assessment_type == 'esm':
                if self.operation_metrics_for_all_time_steps:
                    f.write('param lcia_op {INDICATORS,TECHNOLOGIES,YEARS,YEARS} default 0;\n')
                else:
                    f.write('param lcia_op {INDICATORS,TECHNOLOGIES,YEARS} default 0;\n')
                f.write('param lcia_constr {INDICATORS,TECHNOLOGIES,YEARS} default 0;\n'
                        'param lcia_res {INDICATORS,RESOURCES,YEARS} default 0;\n'
                        'param limit_lcia {INDICATORS,YEARS} default 1e10;\n'
                        'var LCIA_constr {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var LCIA_op {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var LCIA_res {INDICATORS,RESOURCES,YEARS};\n'
                        'var TotalLCIA {INDICATORS,YEARS};\n\n')
            elif assessment_type == 'direct emissions':
                if self.operation_metrics_for_all_time_steps:
                    f.write('param direct_op {INDICATORS,TECHNOLOGIES,YEARS,YEARS} default 0;\n')
                else:
                    f.write('param direct_op {INDICATORS,TECHNOLOGIES,YEARS} default 0;\n')
                f.write('param limit_direct {INDICATORS,YEARS} default 1e10;\n'
                        'var DIRECT_op {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var TotalDIRECT {INDICATORS,YEARS};\n\n')

            if assessment_type == 'esm':
                # Equation of LCIAs variables (construction scaling to F_Mult)
                f.write('# Construction\n'
                        'subject to lcia_constr_calc {id in INDICATORS, i in TECHNOLOGIES, y in YEARS}:\n'
                        '  LCIA_constr[id,i,y] = sum {y_inst in YEARS: y_inst <= y} lcia_constr[id,i,y_inst] '
                        '* F_Mult[i,y_inst] / lifetime[i,y_inst];\n\n')

            # Equation of LCIAs variables (operation scaling to F_Mult_t)
            if self.operation_metrics_for_all_time_steps:
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_op[id,i,y] = sum {{y_inst in YEARS: y_inst <= y}} '
                        f'{metric_type.lower()}_op[id,i,y,y_inst] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t,y,y_inst]);\n\n')
            else:
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_op[id,i,y] = {metric_type.lower()}_op[id,i,y] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t,y]);\n\n')

            if assessment_type == 'esm':
                # Equation of LCIAs variables (resources scaling to F_Mult_t)
                f.write('# Resources\n'
                        'subject to lcia_res_calc {id in INDICATORS, r in RESOURCES, y in YEARS}:\n'
                        '  LCIA_res[id,r,y] = lcia_res[id,r,y] * sum {t in PERIODS} (t_op[t] * F_Mult_t[r,t,y]);\n\n')

            # Equation defining the total LCIA impact (sum over all technologies and resources)
            if assessment_type == 'esm':
                f.write('subject to totalLCIA_calc_r {id in INDICATORS, y in YEARS}:\n'
                        '  TotalLCIA[id,y] = sum {i in TECHNOLOGIES} (LCIA_constr[id,i,y] '
                        '+ LCIA_op[id,i,y]) + sum{r in RESOURCES} (LCIA_res[id,r,y]);\n\n')
            elif assessment_type == 'direct emissions':
                f.write('subject to totalDIRECT_calc_r {id in INDICATORS, y in YEARS}:\n'
                        '  TotalDIRECT[id,y] = sum {i in TECHNOLOGIES} DIRECT_op[id,i,y];\n\n')

            # Equation putting a limit to the total LCIA impact
            f.write(f'subject to total{metric_type}_limit {{id in INDICATORS, y in YEARS}}:\n'
                    f'  Total{metric_type}[id,y] <= limit_{metric_type.lower()}[id,y];\n\n')

            # Declaring the total LCIA amount variables
            for abbrev in list(impact_abbrev.Abbrev):
                f.write(f'var Total{metric_type}_{abbrev}{{y in YEARS}};\n'
                        f'subject to {metric_type}_{abbrev}_cal{{y in YEARS}}:\n'
                        f"  Total{metric_type}_{abbrev}[y] = Total{metric_type}['{abbrev}',y] + TotalCost[y]*1e-6;\n\n")
        else:
            # Declaring the LCIA parameters and variables
            if assessment_type == 'esm':
                f.write('param lcia_constr {INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param lcia_op {INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param lcia_res {INDICATORS,RESOURCES} default 0;\n'
                        'param limit_lcia {INDICATORS} default 1e10;\n'
                        'var LCIA_constr {INDICATORS,TECHNOLOGIES};\n'
                        'var LCIA_op {INDICATORS,TECHNOLOGIES};\n'
                        'var LCIA_res {INDICATORS,RESOURCES};\n'
                        'var TotalLCIA {INDICATORS};\n\n')
            elif assessment_type == 'direct emissions':
                f.write('param direct_op {INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param limit_direct {INDICATORS} default 1e10;\n'
                        'var DIRECT_op {INDICATORS,TECHNOLOGIES};\n'
                        'var TotalDIRECT {INDICATORS};\n\n')

            if assessment_type == 'esm':
                # Equation of LCIAs variables (construction scaling to F_Mult)
                f.write('# Construction\n'
                        'subject to lcia_constr_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                        f'  LCIA_constr[id,i] = lcia_constr[id,i] * F_Mult[i] / lifetime[i];\n\n')

            # Equation of LCIAs variables (operation scaling to F_Mult_t)
            f.write('# Operation\n'
                    f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                    f'  {metric_type}_op[id,i] = {metric_type.lower()}_op[id,i] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t]);\n\n')

            if assessment_type == 'esm':
                # Equation of LCIAs variables (resources scaling to F_Mult_t)
                f.write('# Resources\n'
                        'subject to lcia_res_calc {id in INDICATORS, r in RESOURCES}:\n'
                        '  LCIA_res[id,r] = lcia_res[id,r] * sum {t in PERIODS} (t_op[t] * F_Mult_t[r,t]);\n\n')

            # Equation defining the total LCIA impact (sum over all technologies and resources)
            if assessment_type == 'esm':
                f.write('subject to totalLCIA_calc_r {id in INDICATORS}:\n'
                        '  TotalLCIA[id] = sum {i in TECHNOLOGIES} (LCIA_constr[id,i] '
                        '+ LCIA_op[id,i]) + sum{r in RESOURCES} (LCIA_res[id,r]);\n\n')
            elif assessment_type == 'direct emissions':
                f.write('subject to totalDIRECT_calc_r {id in INDICATORS}:\n'
                        '  TotalDIRECT[id] = sum {i in TECHNOLOGIES} DIRECT_op[id,i];\n\n')

            # Equation putting a limit to the total LCIA impact
            f.write(f'subject to total{metric_type}_limit {{id in INDICATORS}}:\n'
                    f'  Total{metric_type}[id] <= limit_{metric_type.lower()}[id];\n\n')

            # Declaring the total LCIA amount variables
            for abbrev in list(impact_abbrev.Abbrev):
                f.write(f'var Total{metric_type}_{abbrev};\n'
                        f'subject to {metric_type}_{abbrev}_cal:\n'
                        f"  Total{metric_type}_{abbrev} = Total{metric_type}['{abbrev}'] + TotalCost*1e-6;\n\n")
