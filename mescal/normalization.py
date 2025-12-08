import pandas as pd
import ast
from pathlib import Path
import bw2data as bd
from .utils import _short_name_ds_type


def from_str_to_tuple(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a column of strings to tuples

    :param df: dataframe containing the column to be converted
    :param col: column to be converted
    :return: dataframe with the column converted to tuples
    """
    if type(df[col].iloc[0]) is tuple:
        pass
    elif type(df[col].iloc[0]) is str:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    else:
        raise ValueError(f"Unknown type for {col}: {type(df[col].iloc[0])}")

    return df


def restrict_lcia_metrics(
        df: pd.DataFrame,
        lcia_methods: list[str],
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
) -> pd.DataFrame:
    """
    Restrict the dataframe to the LCIA method specified

    :param df: dataframe containing the LCA metrics
    :param lcia_methods: general LCIA method to be used
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :return: dataframe containing the LCA metrics for the specified LCIA method
    """

    df = df[df.apply(lambda x: x.Impact_category[0] in lcia_methods, axis=1)]

    if specific_lcia_categories is not None:
        df = df[df.apply(lambda x: x.Impact_category[-1] in specific_lcia_categories, axis=1)]

    if specific_lcia_abbrev is not None:
        df = df[df.apply(lambda x: x.Abbrev in specific_lcia_abbrev, axis=1)]

    return df


def normalize_lca_metrics(
        self,
        R: pd.DataFrame,
        mip_gap: float,
        impact_abbrev: pd.DataFrame,
        lcia_methods: list[str],
        contrib_processes: pd.DataFrame = None,
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
        assessment_type: str = 'esm',
        path: str = None,
        file_name: str = None,
        metadata: dict = None,
        output: str = 'write',
        skip_normalization: bool = False,
) -> None | pd.DataFrame:
    """
    Create a .dat file containing the normalized LCA metrics for AMPL and a csv file containing the normalization
    factors

    :param R: dataframe containing the LCA indicators results
    :param mip_gap: normalized values that are lower than the MIP gap are set to 0 (to improve numerical stability)
    :param impact_abbrev: dataframe containing the impact categories abbreviations
    :param lcia_methods: list of LCIA methods to be used
    :param contrib_processes: dataframe containing the contribution of processes for each technology/resource and
        impact category. This dataframe must only be provided if assessment_type is 'territorial emissions'. It will
        be used to compute the amount of territorial/abroad impact for each impact category.
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :param assessment_type: type of assessment, can be 'esm' for the full LCA database, 'direct emissions' for the
        computation of direct emissions only, or 'territorial emissions' for the computation of territorial and abroad
        emissions.
    :param path: path to results folder. Default is the results_path_file from the ESM class.
    :param file_name: name of the .dat file. Default is 'techs_lcia' if assessment_type is 'esm',
        'techs_direct' if assessment_type is 'direct emissions', and 'techs_territorial' if assessment_type is
        'territorial emissions'.
    :param metadata: dictionary containing the metadata. Can contain keys 'ecoinvent_version, 'year', 'spatialized',
        'regionalized', 'iam', 'ssp_rcp', 'lcia_method'.
    :param output: if 'write', writes the .dat file in 'path', if 'return', normalizes pandas dataframe, if 'both' does
        both operations.
    :param skip_normalization: if True, skips the normalization step and only writes the .dat file with the original
        values.
    :return: None or the normalized pandas dataframe (depending on the value of 'output')
    """

    if assessment_type == 'territorial emissions' and contrib_processes is None:
        raise ValueError("If assessment_type is 'territorial emissions', contrib_processes must be provided.")

    if assessment_type == 'territorial emissions':
        if 'territorial' not in contrib_processes.columns:
            contrib_processes = self.compute_territorial_impact_scores(contrib_processes)
        contrib_processes = contrib_processes[contrib_processes['territorial'] == True]

        if 'score' in contrib_processes.columns:
            contrib_processes = contrib_processes.rename(columns={'score': 'Value'})
        if 'act_type' in contrib_processes.columns:
            contrib_processes = contrib_processes.rename(columns={'act_type': 'Type'})
        if 'act_name' in contrib_processes.columns:
            contrib_processes = contrib_processes.rename(columns={'act_name': 'Name'})
        if 'impact_category' in contrib_processes.columns:
            contrib_processes = contrib_processes.rename(columns={'impact_category': 'Impact_category'})

    if assessment_type == 'esm':
        metric_type = 'lcia'
    elif assessment_type == 'direct emissions':
        metric_type = 'direct'
    elif assessment_type == 'territorial emissions':
        metric_type = 'territorial'
    else:
        raise ValueError(f"Unknown assessment type: {assessment_type}. Must be 'esm', 'direct emissions' or "
                         f"'territorial emissions'.")

    if metadata is None:
        metadata = {}

    if file_name is None:
        if assessment_type == 'esm':
            file_name = 'techs_lcia'
        elif assessment_type == 'direct emissions':
            file_name = 'techs_direct'
        else:
            file_name = 'techs_territorial'

    if path is None:
        path = self.results_path_file

    R = from_str_to_tuple(R, 'Impact_category')
    impact_abbrev.drop_duplicates(inplace=True)
    impact_abbrev = from_str_to_tuple(impact_abbrev, 'Impact_category')
    if assessment_type == 'territorial emissions':
        contrib_processes = from_str_to_tuple(contrib_processes, 'Impact_category')

    impact_abbrev = restrict_lcia_metrics(
        df=impact_abbrev,
        lcia_methods=lcia_methods,
        specific_lcia_categories=specific_lcia_categories,
        specific_lcia_abbrev=specific_lcia_abbrev,
    )

    if 'Unit' not in impact_abbrev.columns:
        impact_abbrev['Impact_category_unit'] = impact_abbrev['Impact_category'].apply(lambda row: bd.Method(row).metadata['unit'])

    if specific_lcia_categories is not None:
        if len(specific_lcia_categories) > len(impact_abbrev):
            missing_lcia_categories = [cat for cat in specific_lcia_categories if cat not in impact_abbrev['Impact_category'].tolist()]
            raise ValueError(f"The following specified LCIA categories were not found in the impact_abbrev dataframe: {missing_lcia_categories}")

    if specific_lcia_abbrev is not None:
        if len(specific_lcia_abbrev) > len(impact_abbrev):
            missing_lcia_abbrev = [abbrev for abbrev in specific_lcia_abbrev if abbrev not in impact_abbrev['Abbrev'].tolist()]
            raise ValueError(f"The following specified LCIA abbreviations were not found in the impact_abbrev dataframe: {missing_lcia_abbrev}")

    if len(impact_abbrev) == 0:
        raise ValueError("The demanded LCIA categories were not found in the impact_abbrev dataframe.")

    R = pd.merge(R, impact_abbrev, on='Impact_category')
    if assessment_type == 'territorial emissions':
        contrib_processes = pd.merge(contrib_processes, impact_abbrev, on='Impact_category')

    if skip_normalization:
        if assessment_type in ['esm', 'direct emissions']:
            R_scaled = R.copy()
        else:  # assessment_type == 'territorial emissions'
            R_scaled = contrib_processes.copy()
        R_scaled['Value_norm'] = R_scaled['Value']
        norm_unit = ''

    else:
        norm_unit = 'normalized'
        refactor = {}
        R_scaled = R[R['Type'].isin(['Operation', 'Resource'])]
        for unit in R['Unit'].unique():
            # Scale the construction metrics to be at the same order of magnitude as the operation and resource metrics
            lcia_op_max = R[(R['Type'].isin(['Operation', 'Resource'])) & (R['Unit'] == unit)]['Value'].max()
            lcia_constr_max = R[(R['Type'].isin(['Construction', 'Decommission'])) & (R['Unit'] == unit)]['Value'].max()
            refactor[unit] = lcia_op_max / lcia_constr_max
            R_constr_imp = R[(R['Type'].isin(['Construction', 'Decommission'])) & (R['Unit'] == unit)]
            R_constr_imp['Value'] *= refactor[unit]
            R_scaled = pd.concat([R_scaled, R_constr_imp])  # R matrix but with refactor applied to construction metrics
        R_scaled['max_unit'] = R_scaled.groupby('Unit')['Value'].transform('max')

        if assessment_type == 'direct emissions':
            max_per_cat_dict = {}
            max_per_cat = R_scaled[['Abbrev', 'Unit', 'max_unit']].drop_duplicates().reset_index()
            for i in range(len(max_per_cat)):
                max_per_cat_dict[max_per_cat['Unit'][i]] = max_per_cat['max_unit'][i]
            R_scaled = R.copy()
            R_scaled['max_unit'] = R_scaled.apply(lambda x: max_per_cat_dict[x['Unit']], axis=1)

        elif assessment_type == 'territorial emissions':
            max_per_cat_dict = {}
            max_per_cat = R_scaled[['Abbrev', 'Unit', 'max_unit']].drop_duplicates().reset_index()
            R_scaled = contrib_processes[contrib_processes['Type'].isin(['Operation', 'Resource'])]
            for unit in contrib_processes['Unit'].unique():
                # Scale the construction metrics to be at the same order of magnitude as the operation and resource metrics
                R_constr_imp = contrib_processes[(contrib_processes['Type'].isin(['Construction', 'Decommission'])) & (contrib_processes['Unit'] == unit)]
                R_constr_imp['Value'] *= refactor[unit]
                R_scaled = pd.concat([R_scaled, R_constr_imp])  # R matrix but with refactor applied to construction metrics
            for i in range(len(max_per_cat)):
                max_per_cat_dict[max_per_cat['Unit'][i]] = max_per_cat['max_unit'][i]
            R_scaled['max_unit'] = R_scaled.apply(lambda x: max_per_cat_dict[x['Unit']], axis=1)

        R_scaled['Value_norm'] = R_scaled['Value'] / R_scaled['max_unit']

        if assessment_type in ['esm', 'territorial emissions']:
            R_scaled_constr = R_scaled[R_scaled['Type'].isin(['Construction', 'Decommission'])]
            R_scaled_op = R_scaled[R_scaled['Type'].isin(['Operation', 'Resource'])]
            R_scaled_op['Value_norm'] = R_scaled_op['Value_norm'].apply(lambda x: 0 if abs(x) < mip_gap else x)
            R_scaled_constr['Value_norm'] = R_scaled_constr.apply(lambda x: 0 if abs(x['Value_norm']) < mip_gap else x['Value_norm'] / refactor[x['Unit']], axis=1)
            R_scaled = pd.concat([R_scaled_op, R_scaled_constr])
        else:  # assessment_type == 'direct emissions'
            R_scaled['Value_norm'] = R_scaled['Value_norm'].apply(lambda x: 0 if abs(x) < mip_gap else x)

    if (output == 'write') | (output == 'both'):

        Path(path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist

        with open(f'{path}{file_name}.dat', 'w') as f:

            # Write metadata at the beginning of the file
            if 'ecoinvent_version' in metadata:
                f.write(f"# Ecoinvent version: {metadata['ecoinvent_version']}\n")
            if 'spatialized' in metadata:
                f.write(f"# Spatialized database: {metadata['spatialized']}\n")
            if 'regionalized' in metadata:
                f.write(f"# Regionalized database: {metadata['regionalized']}\n")
            if 'year' in metadata:
                f.write(f"# Selected year(s) in premise: {metadata['year']}\n")
            if 'iam' in metadata:
                f.write(f"# Selected IAM in premise: {metadata['iam']}\n")
            if 'ssp_rcp' in metadata:
                f.write(f"# Selected SSP-RCP scenario in premise: {metadata['ssp_rcp']}\n")
            if 'lcia_method' in metadata:
                f.write(f"# LCIA method: {metadata['lcia_method']}\n")
            f.write("\n")

            if assessment_type == 'esm':
                # Set of LCA indicators and units
                f.write(f"set INDICATORS := {' '.join(R_scaled['Abbrev'].unique())};\n\n")

            # Declare the LCA indicators values
            for i in range(len(R_scaled)):
                # Declaring the LCIA parameters
                if self.pathway:
                    if self.operation_metrics_for_all_time_steps:
                        f.write(
                            f"let {metric_type}_{_short_name_ds_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}',"
                            f"'{R_scaled.Name.iloc[i]}',{R_scaled.Year.iloc[i]},{R_scaled.Year_inst.iloc[i]}] "
                            f":= {R_scaled.Value_norm.iloc[i]}; #{norm_unit} {R_scaled.Unit.iloc[i]}\n")
                    else:
                        f.write(
                            f"let {metric_type}_{_short_name_ds_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}',"
                            f"'{R_scaled.Name.iloc[i]}',{R_scaled.Year.iloc[i]}] := {R_scaled.Value_norm.iloc[i]}; "
                            f"#{norm_unit} {R_scaled.Unit.iloc[i]}\n")
                else:
                    f.write(f"let {metric_type}_{_short_name_ds_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}','{R_scaled.Name.iloc[i]}'] "
                            f":= {R_scaled.Value_norm.iloc[i]}; #{norm_unit} {R_scaled.Unit.iloc[i]}\n")

        if not skip_normalization:
            # To come back to the original values, we save the maximum value of each unit
            if assessment_type == 'esm':
                R_scaled[['Abbrev', 'Unit', 'max_unit']].drop_duplicates().to_csv(f'{path}{file_name}_max.csv', index=False)

        if output == 'both':
            return R_scaled

    elif output == 'return':
        return R_scaled

    else:
        raise ValueError(f"The output parameter must be either 'write', 'return' or 'both'")