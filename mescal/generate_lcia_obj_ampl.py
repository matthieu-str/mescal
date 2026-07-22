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
                f.write(f'param {metric_type.lower()}_op {{YEARS,INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_constr {{YEARS,INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_decom {{YEARS,INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_res {{YEARS,INDICATORS,RESOURCES}} default 0;\n'
                        f'param limit_{metric_type.lower()}_year {{YEARS,INDICATORS}} default Infinity;\n'
                        f'param limit_{metric_type.lower()} {{INDICATORS}} default Infinity;\n'
                        f'var {metric_type}_constr {{PHASE,INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_decom {{PHASE,INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_op {{PHASE,INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_res {{PHASE,INDICATORS,RESOURCES}};\n'
                        f'var Phase{metric_type} {{PHASE,INDICATORS}};\n'
                        f'var Total{metric_type} {{INDICATORS}};\n\n')

            elif assessment_type == 'direct emissions':
                f.write('param direct_op {YEARS,INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param limit_direct_year {YEARS,INDICATORS} default Infinity;\n'
                        'param limit_direct {INDICATORS} default Infinity;\n'
                        'var DIRECT_op {PHASE,INDICATORS,TECHNOLOGIES};\n'
                        'var PhaseDIRECT {PHASE,INDICATORS};\n'
                        'var TotalDIRECT {INDICATORS};\n\n')

            if assessment_type == 'territorial emissions':
                # Abroad emissions parameters and variables
                f.write('param limit_abroad_year {YEARS,INDICATORS} default Infinity;\n'
                        'param limit_abroad {INDICATORS} default Infinity;\n'
                        'var ABROAD_constr {PHASE,INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_decom {PHASE,INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_op {PHASE,INDICATORS,TECHNOLOGIES};\n'
                        'var ABROAD_res {PHASE,INDICATORS,RESOURCES};\n'
                        'var PhaseABROAD {PHASE,INDICATORS};\n'
                        'var TotalABROAD {INDICATORS};\n\n')

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of infrastructure variables (construction and decommission impacts scaled with the installed capacity)
                f.write('# Construction\n'
                        f'subject to {metric_type.lower()}_constr_calc {{p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, i in TECHNOLOGIES}}:\n'
                        f'  {metric_type}_constr[p,id,i] =\n'
                        f'    sum {{p_inst in PHASE_WND union PHASE_UP_TO,\n'
                        f'      ys_inst in PHASE_START[p_inst],\n'
                        f'      ye_inst in PHASE_STOP[p_inst]:\n'
                        f'      years_active[i,p_inst,p] > 0}}\n'
                        f'      ({metric_type.lower()}_constr[ys_inst,id,i] + {metric_type.lower()}_constr[ye_inst,id,i]) / 2\n'
                        f'      * F_new[p_inst,i] * years_active[i, p_inst, p] / ((lifetime[ys_inst,i] + lifetime[ye_inst,i]) / 2);\n\n')

                f.write('# Decommission\n'
                        f'subject to {metric_type.lower()}_decom_calc {{p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, i in TECHNOLOGIES}}:\n'
                        f'  {metric_type}_decom[p,id,i] =\n'
                        f'    sum {{p_inst in PHASE_WND union PHASE_UP_TO,\n'
                        f'      ys_inst in PHASE_START[p_inst],\n'
                        f'      ye_inst in PHASE_STOP[p_inst]:\n'
                        f'      years_active[i,p_inst,p] > 0}}\n'
                        f'      ({metric_type.lower()}_decom[ys_inst,id,i] + {metric_type.lower()}_decom[ye_inst,id,i]) / 2\n'
                        f'      * F_new[p_inst,i] * years_active[i, p_inst, p] / ((lifetime[ys_inst,i] + lifetime[ye_inst,i]) / 2);\n\n')

            # Equation of operation variables (operation impacts scaled with the annual production)
            f.write('# Operation\n'
                    f'subject to {metric_type.lower()}_op_calc {{p in PHASE_WND union PHASE_UP_TO, y_start in PHASE_START[p], y_stop in PHASE_STOP[p], id in INDICATORS, i in TECHNOLOGIES}}:\n'
                    f'  {metric_type}_op[p,id,i] =\n'
                    f'    ({metric_type.lower()}_op[y_start,id,i] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[y_start,i,t])\n'
                    f'    + {metric_type.lower()}_op[y_stop,id,i] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[y_stop,i,t])) / 2\n'
                    f'    * t_phase;\n\n')

            if assessment_type in ['esm', 'territorial emissions']:
                # Equation of resource variables (resources impacts scaled with the annual usage)
                f.write('# Resources\n'
                        f'subject to {metric_type.lower()}_res_calc {{p in PHASE_WND union PHASE_UP_TO, y_start in PHASE_START[p], y_stop in PHASE_STOP[p], id in INDICATORS, r in RESOURCES}}:\n'
                        f'  {metric_type}_res[p,id,r] =\n'
                        f'    ({metric_type.lower()}_res[y_start,id,r] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[y_start,r,t])\n'
                        f'    + {metric_type.lower()}_res[y_stop,id,r] * sum {{t in PERIODS}} (t_op[t] * F_Mult_t[y_stop,r,t])) / 2\n'
                        f'    * t_phase;\n\n')

            if assessment_type == 'territorial emissions':
                # Equations of abroad emissions
                f.write('# Abroad impacts\n'
                        'subject to abroad_constr_calc {p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_constr[p,id,i] = LCIA_constr[p,id,i] - TERRITORIAL_constr[p,id,i];\n'
                        '\n'
                        'subject to abroad_decom_calc {p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_decom[p,id,i] = LCIA_decom[p,id,i] - TERRITORIAL_decom[p,id,i];\n'
                        '\n'
                        'subject to abroad_op_calc {p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, i in TECHNOLOGIES}:\n'
                        '  ABROAD_op[p,id,i] = LCIA_op[p,id,i] - TERRITORIAL_op[p,id,i];\n'
                        '\n'
                        'subject to abroad_res_calc {p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, r in RESOURCES}:\n'
                        '  ABROAD_res[p,id,r] = LCIA_res[p,id,r] - TERRITORIAL_res[p,id,r];\n\n')

            # Equation defining the total impact (sum over all technologies and resources)
            if assessment_type in ['esm', 'territorial emissions']:
                if assessment_type == 'territorial emissions':
                    metric_type_list = [metric_type, 'ABROAD']
                else:
                    metric_type_list = [metric_type]
                for metric in metric_type_list:
                    f.write(f'subject to phase{metric}_calc_r {{p in PHASE_WND union PHASE_UP_TO, id in INDICATORS}}:\n'
                            f'  Phase{metric}[p,id] = sum {{i in TECHNOLOGIES}} ({metric}_constr[p,id,i] + {metric}_decom[p,id,i]\n'
                            f'+ {metric}_op[p,id,i]) + sum{{r in RESOURCES}} ({metric}_res[p,id,r]);\n\n'
                            f'subject to total{metric}_calc_r {{id in INDICATORS}}:\n'
                            f'  Total{metric}[id] = sum {{p in PHASE_WND union PHASE_UP_TO}} Phase{metric}[p,id];\n\n')
            elif assessment_type == 'direct emissions':
                f.write('subject to phaseDIRECT_calc_r {p in PHASE_WND union PHASE_UP_TO, id in INDICATORS}:\n'
                        '  PhaseDIRECT[p,id] = sum {i in TECHNOLOGIES} DIRECT_op[p,id,i];\n\n'
                        'subject to totalDIRECT_calc_r {id in INDICATORS}:\n'
                        '  TotalDIRECT[id] = sum {p in PHASE_WND union PHASE_UP_TO} PhaseDIRECT[p,id];\n\n')

            # Equation putting a limit to the total impact
            if assessment_type == 'territorial emissions':
                metric_type_list = [metric_type, 'ABROAD']
            else:
                metric_type_list = [metric_type]
            for metric in metric_type_list:
                f.write(f'subject to phase{metric}_limit {{p in PHASE_WND union PHASE_UP_TO, id in INDICATORS, y_start in PHASE_START[p], y_stop in PHASE_STOP[p]}}:\n'
                        f'  Phase{metric}[p,id] <= t_phase * (limit_{metric.lower()}_year[y_start,id] + limit_{metric.lower()}_year[y_stop,id]) / 2;\n\n'
                        f'subject to total{metric}_limit {{id in INDICATORS}}:\n'
                        f'  Total{metric}[id] <= limit_{metric.lower()}[id];\n\n')

                # Declaring the total amount variables
                for abbrev in list(impact_abbrev.Abbrev):
                    f.write(f'var Total{metric}_{abbrev};\n'
                            f'subject to {metric}_{abbrev}_cal:\n'
                            f"  Total{metric}_{abbrev} = Total{metric}['{abbrev}'] + TotalTransitionCost*1e-6;\n\n")
        else:
            # Declaring the parameters and variables
            if assessment_type in ['esm', 'territorial emissions']:
                f.write(f'param {metric_type.lower()}_constr {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_decom {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_op {{INDICATORS,TECHNOLOGIES}} default 0;\n'
                        f'param {metric_type.lower()}_res {{INDICATORS,RESOURCES}} default 0;\n'
                        f'param limit_{metric_type.lower()} {{INDICATORS}} default Infinity;\n'
                        f'var {metric_type}_constr {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_decom {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_op {{INDICATORS,TECHNOLOGIES}};\n'
                        f'var {metric_type}_res {{INDICATORS,RESOURCES}};\n'
                        f'var Total{metric_type} {{INDICATORS}};\n\n')

            elif assessment_type == 'direct emissions':
                f.write('param direct_op {INDICATORS,TECHNOLOGIES} default 0;\n'
                        'param limit_direct {INDICATORS} default Infinity;\n'
                        'var DIRECT_op {INDICATORS,TECHNOLOGIES};\n'
                        'var TotalDIRECT {INDICATORS};\n\n')

            if assessment_type == 'territorial emissions':
                # Abroad emissions parameters and variables
                f.write('param limit_abroad {INDICATORS} default Infinity;\n'
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
