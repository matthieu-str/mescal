import pandas as pd
import ast
from pathlib import Path


@staticmethod
def _tech_type(tech: str) -> str:
    """
    Returns the short name of the technology type

    :param tech: type of technology
    :return: short name of the technology type
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

    :param lcia_method: name of the LCIA method
    :return: short name for the LCIA method
    """
    if lcia_method == 'IMPACT World+ Damage 2.0.1':
        return 'endpoint'
    elif lcia_method == 'IMPACT World+ Damage 2.0.1 - Total only':
        return 'endpoint_tot'
    elif lcia_method == 'IMPACT World+ Midpoint 2.0.1':
        return 'midpoint'
    elif lcia_method == 'IMPACT World+ Footprint 2.0.1':
        return 'footprint'
    if lcia_method == 'IMPACT World+ Damage 2.0.1_regionalized':
        return 'endpoint_reg'
    elif lcia_method == 'IMPACT World+ Damage 2.0.1_regionalized - Total only':
        return 'endpoint_tot_reg'
    elif lcia_method == 'IMPACT World+ Midpoint 2.0.1_regionalized':
        return 'midpoint_reg'
    elif lcia_method == 'IMPACT World+ Footprint 2.0.1_regionalized':
        return 'footprint_reg'
    elif lcia_method == 'PB LCIA':
        return 'pb_lcia'
    else:
        raise ValueError(f"Unknown LCIA method: {lcia_method}")


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
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
        assessment_type: str = 'esm',
        max_per_cat: pd.DataFrame = None,
        path: str = 'results/',
        file_name: str = 'techs_lcia',
        metadata: dict = None,
        output: str = 'write',
        skip_normalization: bool = False,
) -> None | pd.DataFrame:
    """
    Create a .dat file containing the normalized LCA metrics for AMPL and a csv file containing the normalization
    factors

    :param path: path to results folder
    :param file_name: name of the .dat file
    :param R: dataframe containing the LCA indicators results
    :param mip_gap: normalized values that are lower than the MIP gap are set to 0 (to improve numerical stability)
    :param lcia_methods: LCIA method to be used
    :param specific_lcia_categories: specific LCIA categories to be used
    :param specific_lcia_abbrev: specific LCIA abbreviations to be used
    :param assessment_type: type of assessment, can be 'esm' for the full LCA database, or 'direct emissions' for the
        computation of territorial emissions only
    :param max_per_cat: dataframe containing the maximum value of each impact unit, needed if assessment_type is 'direct
        emissions'
    :param impact_abbrev: dataframe containing the impact categories abbreviations
    :param metadata: dictionary containing the metadata. Can contain keys 'ecoinvent_version, 'year', 'spatialized',
        'regionalized', 'iam', 'ssp_rcp', 'lcia_method'.
    :param output: if 'write', writes the .dat file in 'path', if 'return', normalizes pandas dataframe, if 'both' does
        both operations.
    :param skip_normalization: if True, skips the normalization step and only writes the .dat file with the original
        values.
    :return: None or the normalized pandas dataframe (depending on the value of 'output')
    """

    if assessment_type == 'direct emissions' and max_per_cat is None:
        raise ValueError("If assessment_type is 'direct emissions', max_per_cat must be provided. Run this method with "
                         "assessment_type='esm' first to get the max_per_cat dataframe.")

    if assessment_type == 'esm':
        metric_type = 'lcia'
    elif assessment_type == 'direct emissions':
        metric_type = 'direct'
    else:
        raise ValueError(f"Unknown assessment type: {assessment_type}. Must be 'esm' or 'direct emissions'.")

    if metadata is None:
        metadata = {}

    R = from_str_to_tuple(R, 'Impact_category')
    impact_abbrev = from_str_to_tuple(impact_abbrev, 'Impact_category')

    impact_abbrev = restrict_lcia_metrics(
        df=impact_abbrev,
        lcia_methods=lcia_methods,
        specific_lcia_categories=specific_lcia_categories,
        specific_lcia_abbrev=specific_lcia_abbrev,
    )

    R = pd.merge(R, impact_abbrev, on='Impact_category')

    if skip_normalization:
        R_scaled = R.copy()
        R_scaled['Value_norm'] = R_scaled['Value']
        norm_unit = ''

    else:
        norm_unit = 'normalized'
        if assessment_type == 'esm':
            refactor = {}
            R_scaled = R[R['Type'] != 'Construction']
            for unit in R['Unit'].unique():
                # Scale the construction metrics to be at the same order of magnitude as the operation and resource metrics
                lcia_op_max = R[((R['Type'] == 'Operation') |
                                 (R['Type'] == 'Resource')) & (R['Unit'] == unit)]['Value'].max()
                lcia_constr_max = R[(R['Type'] == 'Construction') & (R['Unit'] == unit)]['Value'].max()
                refactor[unit] = lcia_op_max / lcia_constr_max
                R_constr_imp = R[(R['Type'] == 'Construction') & (R['Unit'] == unit)]
                R_constr_imp['Value'] *= refactor[unit]
                R_scaled = pd.concat([R_scaled, R_constr_imp])  # R matrix but with refactor applied to construction metrics
                R_scaled['max_unit'] = R_scaled.groupby('Unit')['Value'].transform('max')

        else:  # assessment_type == 'direct emissions'
            refactor = None  # not needed for direct emissions as they are for operation datasets only
            max_per_cat_dict = {}
            for i in range(len(max_per_cat)):
                max_per_cat_dict[max_per_cat['Unit'][i]] = max_per_cat['max_unit'][i]
            R_scaled = R.copy()
            R_scaled['max_unit'] = R_scaled.apply(lambda x: max_per_cat_dict[x['Unit']], axis=1)

        R_scaled['Value_norm'] = R_scaled['Value'] / R_scaled['max_unit']
        if assessment_type == 'esm':
            R_scaled_constr = R_scaled[R_scaled['Type'] == 'Construction']
            R_scaled_op = R_scaled[R_scaled['Type'] != 'Construction']
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
                            f"let {metric_type}_{self._tech_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}',"
                            f"'{R_scaled.Name.iloc[i]}',{R_scaled.Year.iloc[i]},{R_scaled.Year_inst.iloc[i]}] "
                            f":= {R_scaled.Value_norm.iloc[i]}; #{norm_unit} {R_scaled.Unit.iloc[i]}\n")
                    else:
                        f.write(
                            f"let {metric_type}_{self._tech_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}',"
                            f"'{R_scaled.Name.iloc[i]}',{R_scaled.Year.iloc[i]}] := {R_scaled.Value_norm.iloc[i]}; "
                            f"#{norm_unit} {R_scaled.Unit.iloc[i]}\n")
                else:
                    f.write(f"let {metric_type}_{self._tech_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}','{R_scaled.Name.iloc[i]}'] "
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
