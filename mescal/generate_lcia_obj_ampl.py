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
        path: str = None,
        file_name: str = None,
        metadata: dict = None,
        energyscope_version: str = 'epfl',
) -> None:
    """
    Create an AMPL mod file containing the LCIA equations. This method has been specifically designed for the
    EnergyScope model. Currently, it supports the 'epfl' and 'core' versions of EnergyScope.

    :param impact_abbrev: dataframe containing the impact abbreviations of the LCIA method
    :param lcia_methods: LCIA methods to be used
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :param assessment_type: type of assessment, can be 'esm' for the full LCA database, 'direct emissions' for the
        computation of direct emissions only, or 'territorial emissions' for territorial and abroad emissions.
        Default is 'esm'.
    :param path: path where the mod file will be saved. Default is results_path_file from the ESM class.
    :param file_name: name of the .mod file. Default is 'objectives' if assessment_type is 'esm',
        'objectives_direct' if assessment_type is 'direct emissions', and 'objectives_territorial' if assessment_type is
        'territorial emissions'.
    :param metadata: dictionary containing the metadata to be written at the beginning of the file
    :param energyscope_version: version of EnergyScope model used, can be 'epfl' or 'core'
    :return: None (writes the file)
    """

    if assessment_type == 'esm':
        metric_type = 'LCIA'
    elif assessment_type == 'direct emissions':
        metric_type = 'DIRECT'
    elif assessment_type == 'territorial emissions':
        metric_type = 'TERRITORIAL'
    else:
        raise ValueError(f"Unknown assessment type: {assessment_type}. Must be 'esm' or 'direct emissions'.")

    if metadata is None:
        metadata = {}

    if file_name is None:
        if assessment_type == 'esm':
            file_name = 'objectives'
        elif assessment_type == 'direct emissions':
            file_name = 'objectives_direct'
        else:  # assessment_type == 'territorial emissions'
            file_name = 'objectives_territorial'

    if path is None:
        path = self.results_path_file

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
            # Declaring the parameters and variables
            if assessment_type in ['esm', 'territorial emissions']:
                if self.operation_metrics_for_all_time_steps:
                    f.write(f'param {metric_type.lower()}_op {{INDICATORS,TECHNOLOGIES,YEARS,YEARS}} default 0;\n')
                else:
                    f.write(f'param {metric_type.lower()}_op {{INDICATORS,TECHNOLOGIES,YEARS}} default 0;\n')
                f.write(f'param {metric_type.lower()}_constr {{INDICATORS,TECHNOLOGIES,YEARS}} default 0;\n'
                        f'param {metric_type.lower()}_decom {{INDICATORS,TECHNOLOGIES,YEARS}} default 0;\n'
                        f'param {metric_type.lower()}_res {{INDICATORS,RESOURCES,YEARS}} default 0;\n'
                        f'param limit_{metric_type.lower()} {{INDICATORS,YEARS}} default 1e10;\n'
                        f'var {metric_type}_constr {{INDICATORS,TECHNOLOGIES,YEARS}};\n'
                        f'var {metric_type}_decom {{INDICATORS,TECHNOLOGIES,YEARS}};\n'
                        f'var {metric_type}_op {{INDICATORS,TECHNOLOGIES,YEARS}};\n'
                        f'var {metric_type}_res {{INDICATORS,RESOURCES,YEARS}};\n'
                        f'var Total{metric_type} {{INDICATORS,YEARS}};\n\n')

            elif assessment_type == 'direct emissions':
                if self.operation_metrics_for_all_time_steps:
                    f.write('param direct_op {INDICATORS,TECHNOLOGIES,YEARS,YEARS} default 0;\n')
                else:
                    f.write('param direct_op {INDICATORS,TECHNOLOGIES,YEARS} default 0;\n')
                f.write('param limit_direct {INDICATORS,YEARS} default 1e10;\n'
                        'var DIRECT_op {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var TotalDIRECT {INDICATORS,YEARS};\n\n')

            if assessment_type == 'territorial emissions':
                # Abroad emissions parameters and variables
                f.write('param limit_abroad {INDICATORS,YEARS} default 1e10;\n'
                        'var ABROAD_constr {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var ABROAD_decom {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var ABROAD_op {INDICATORS,TECHNOLOGIES,YEARS};\n'
                        'var ABROAD_res {INDICATORS,RESOURCES,YEARS};\n'
                        'var TotalABROAD {INDICATORS,YEARS};\n\n')

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of infrastructure variables (construction and decommission impacts scaled with the installed capacity)
                f.write('# Construction\n'
                        f'subject to {metric_type.lower()}_constr_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_constr[id,i,y] = sum {{y_inst in YEARS: y_inst <= y}} {metric_type.lower()}_constr[id,i,y_inst] '
                        f'* F_Mult[i,y_inst] / lifetime[i,y_inst];\n\n')

                f.write('# Decommission\n'
                        f'subject to {metric_type.lower()}_decom_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_decom[id,i,y] = sum {{y_inst in YEARS: y_inst <= y}} {metric_type.lower()}_decom[id,i,y_inst] '
                        f'* F_Mult[i,y_inst] / lifetime[i,y_inst];\n\n')

            # Equation of operation variables (operation impacts scaled with the annual production)
            if self.operation_metrics_for_all_time_steps:
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_op[id,i,y] = sum {{y_inst in YEARS: y_inst <= y}} '
                        f'{metric_type.lower()}_op[id,i,y,y_inst] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t,y,y_inst]);\n\n')
            else:
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES, y in YEARS}}:\n'
                        f'  {metric_type}_op[id,i,y] = {metric_type.lower()}_op[id,i,y] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t,y]);\n\n')

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of resource variables (resources impacts scaled with the annual usage)
                f.write('# Resources\n'
                        f'subject to {metric_type.lower()}_res_calc {{id in INDICATORS, r in RESOURCES, y in YEARS}}:\n'
                        f'  {metric_type}_res[id,r,y] = {metric_type.lower()}_res[id,r,y] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[r,t,y]);\n\n')

            if assessment_type == 'territorial emissions':
                # Equations of abroad emissions
                f.write('# Abroad impacts\n'
                        'subject to abroad_constr_calc {id in INDICATORS, i in TECHNOLOGIES y in YEARS}:\n'
                        '  ABROAD_constr[id,i,y] = LCIA_constr[id,i,y] - TERRITORIAL_constr[id,i,y];\n'
                        '\n'
                        'subject to abroad_decom_calc {id in INDICATORS, i in TECHNOLOGIES y in YEARS}:\n'
                        '  ABROAD_decom[id,i,y] = LCIA_decom[id,i,y] - TERRITORIAL_decom[id,i,y];\n'
                        '\n'
                        'subject to abroad_op_calc {id in INDICATORS, i in TECHNOLOGIES y in YEARS}:\n'
                        '  ABROAD_op[id,i,y] = LCIA_op[id,i,y] - TERRITORIAL_op[id,i,y];\n'
                        '\n'
                        'subject to abroad_res_calc {id in INDICATORS, r in RESOURCES y in YEARS}:\n'
                        '  ABROAD_res[id,r,y] = LCIA_res[id,r,y] - TERRITORIAL_res[id,r,y];\n\n')

            # Equation defining the total impact (sum over all technologies and resources)
            if assessment_type in ['esm', 'territorial emissions']:
                if assessment_type == 'territorial emissions':
                    metric_type_list = [metric_type, 'ABROAD']
                else:
                    metric_type_list = [metric_type]
                for metric in metric_type_list:
                    f.write(f'subject to total{metric}_calc_r {{id in INDICATORS, y in YEARS}}:\n'
                            f'  Total{metric}[id,y] = sum {{i in TECHNOLOGIES}} ({metric}_constr[id,i,y] + {metric}_decom[id,i,y] '
                            f'+ {metric}_op[id,i,y]) + sum{{r in RESOURCES}} ({metric}_res[id,r,y]);\n\n')
            elif assessment_type == 'direct emissions':
                f.write('subject to totalDIRECT_calc_r {id in INDICATORS, y in YEARS}:\n'
                        '  TotalDIRECT[id,y] = sum {i in TECHNOLOGIES} DIRECT_op[id,i,y];\n\n')

            # Equation putting a limit to the total impact
            if assessment_type == 'territorial emissions':
                metric_type_list = [metric_type, 'ABROAD']
            else:
                metric_type_list = [metric_type]
            for metric in metric_type_list:
                f.write(f'subject to total{metric}_limit {{id in INDICATORS, y in YEARS}}:\n'
                        f'  Total{metric}[id,y] <= limit_{metric.lower()}[id,y];\n\n')

                # Declaring the total amount variables
                for abbrev in list(impact_abbrev.Abbrev):
                    f.write(f'var Total{metric}_{abbrev}{{y in YEARS}};\n'
                            f'subject to {metric}_{abbrev}_cal{{y in YEARS}}:\n'
                            f"  Total{metric}_{abbrev}[y] = Total{metric}['{abbrev}',y] + TotalCost[y]*1e-6;\n\n")
        else:
            # Declaring the parameters and variables
            if assessment_type in ['esm', 'territorial emissions']:
                f.write(f'param {metric_type.lower()}_constr {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_decom {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_op {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_res {{INDICATORS,RESOURCES}} default 0;\n'
                        f'param limit_{metric_type.lower()} {{INDICATORS}} default 1e10;\n'
                        f'var {metric_type}_constr {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_decom {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_op {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_res {{INDICATORS,RESOURCES}};\n'
                        f'var Total{metric_type} {{INDICATORS}};\n\n')

            elif assessment_type == 'direct emissions':
                f.write('param direct_op {INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param limit_direct {INDICATORS} default 1e10;\n'
                        'var DIRECT_op {INDICATORS,TECHNOLOGIES};\n'
                        'var TotalDIRECT {INDICATORS};\n\n')

            if assessment_type == 'territorial emissions':
                # Abroad emissions parameters and variables
                f.write('param limit_abroad {INDICATORS} default 1e10;\n'
                        'var ABROAD_constr {INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_decom {INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_op {INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_res {INDICATORS,RESOURCES};\n'
                        'var TotalABROAD {INDICATORS};\n\n')

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of infrastructure variables (construction and decommission impacts scaled with the installed capacity)
                if energyscope_version == 'epfl':
                    f.write('# Construction\n'
                            f'subject to {metric_type.lower()}_constr_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                            f'  {metric_type}_constr[id,i] = {metric_type.lower()}_constr[id,i] * F_Mult[i] / lifetime[i];\n\n')

                    f.write('# Decommission\n'
                            f'subject to {metric_type.lower()}_decom_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                            f'  {metric_type}_decom[id,i] = {metric_type.lower()}_decom[id,i] * F_Mult[i] / lifetime[i];\n\n')

                elif energyscope_version == 'core':
                    f.write('# Construction\n'
                            f'subject to {metric_type.lower()}_constr_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                            f'  {metric_type}_constr[id,i] = {metric_type.lower()}_constr[id,i] * F[i] / lifetime[i];\n\n')

                    f.write('# Decommission\n'
                            f'subject to {metric_type.lower()}_decom_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                            f'  {metric_type}_decom[id,i] = {metric_type.lower()}_decom[id,i] * F[i] / lifetime[i];\n\n')

                else:
                    raise ValueError(f"Unknown energyscope_version: {energyscope_version}. Only 'epfl' and 'core' are "
                                     f"supported.")

            # Equation of operation variables (operation impacts scaled with the annual production)
            if energyscope_version == 'epfl':
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                        f'  {metric_type}_op[id,i] = {metric_type.lower()}_op[id,i] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[i,t]);\n\n')

            elif energyscope_version == 'core':
                f.write('# Operation\n'
                        f'subject to {metric_type.lower()}_op_calc {{id in INDICATORS, i in TECHNOLOGIES}}:\n'
                        f'  {metric_type}_op[id,i] = {metric_type.lower()}_op[id,i] * sum {{t in PERIODS, h in HOUR_OF_PERIOD [t], '
                        f'td in TYPICAL_DAY_OF_PERIOD [t]}} (t_op[h,td] * F_t[i,h,td]);\n\n')

            else:
                raise ValueError(f"Unknown energyscope_version: {energyscope_version}. Only 'epfl' and 'core' are "
                                 f"supported.")

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of resource variables (resources impacts scaled with the annual usage)
                if energyscope_version == 'epfl':
                    f.write('# Resources\n'
                            f'subject to {metric_type.lower()}_res_calc {{id in INDICATORS, r in RESOURCES}}:\n'
                            f'  {metric_type}_res[id,r] = {metric_type.lower()}_res[id,r] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[r,t]);\n\n')

                elif energyscope_version == 'core':
                    f.write('# Resources\n'
                            f'subject to {metric_type.lower()}_res_calc {{id in INDICATORS, r in RESOURCES}}:\n'
                            f'  {metric_type}_res[id,r] = {metric_type.lower()}_res[id,r] * sum {{t in PERIODS, h in HOUR_OF_PERIOD [t], '
                            f'td in TYPICAL_DAY_OF_PERIOD [t]}} (t_op[h,td] * F_t[r,h,td]);\n\n')

                else:
                    raise ValueError(f"Unknown energyscope_version: {energyscope_version}. Only 'epfl' and 'core' are "
                                     f"supported.")

            if assessment_type == 'territorial emissions':
                # Equations of abroad emissions
                f.write('# Abroad impacts\n'
                        'subject to abroad_constr_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_constr[id,i] = LCIA_constr[id,i] - TERRITORIAL_constr[id,i];\n'
                        '\n'
                        'subject to abroad_decom_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_decom[id,i] = LCIA_decom[id,i] - TERRITORIAL_decom[id,i];\n'
                        '\n'
                        'subject to abroad_op_calc {id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_op[id,i] = LCIA_op[id,i] - TERRITORIAL_op[id,i];\n'
                        '\n'
                        'subject to abroad_res_calc {id in INDICATORS, r in RESOURCES}:\n'
                        '  ABROAD_res[id,r] = LCIA_res[id,r] - TERRITORIAL_res[id,r];\n\n')

            # Equation defining the total impact (sum over all technologies and resources)
            if assessment_type in ['esm', 'territorial emissions']:
                if assessment_type == 'territorial emissions':
                    metric_type_list = [metric_type, 'ABROAD']
                else:
                    metric_type_list = [metric_type]
                for metric in metric_type_list:
                    f.write(f'subject to total{metric}_calc_r {{id in INDICATORS}}:\n'
                            f'  Total{metric}[id] = sum {{i in TECHNOLOGIES}} ({metric}_constr[id,i] + {metric}_decom[id,i] '
                            f'+ {metric}_op[id,i]) + sum{{r in RESOURCES}} ({metric}_res[id,r]);\n\n')

            elif assessment_type == 'direct emissions':
                f.write('subject to totalDIRECT_calc_r {id in INDICATORS}:\n'
                        '  TotalDIRECT[id] = sum {i in TECHNOLOGIES} DIRECT_op[id,i];\n\n')

            # Equation putting a limit to the total impact
            if assessment_type == 'territorial emissions':
                metric_type_list = [metric_type, 'ABROAD']
            else:
                metric_type_list = [metric_type]
            for metric in metric_type_list:
                f.write(f'subject to total{metric}_limit {{id in INDICATORS}}:\n'
                        f'  Total{metric}[id] <= limit_{metric.lower()}[id];\n\n')

                # Declaring the total amount variables
                for abbrev in list(impact_abbrev.Abbrev):
                    f.write(f'var Total{metric}_{abbrev};\n'
                            f'subject to {metric}_{abbrev}_cal:\n'
                            f"  Total{metric}_{abbrev} = Total{metric}['{abbrev}'] + TotalCost*1e-6;\n\n")
