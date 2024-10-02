import pandas as pd
import ast
from pathlib import Path


def tech_type(tech: str) -> str:
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
        lcia_method: str
) -> pd.DataFrame:
    """
    Restrict the dataframe to the LCIA method specified

    :param df: dataframe containing the LCA metrics
    :param lcia_method: LCIA method to be used
    :return: dataframe containing the LCA metrics for the specified LCIA method
    """
    if 'Total only' in lcia_method:
        method_name = lcia_method.split(' - ')[0]
        df = df[df.apply(lambda x: x.Impact_category[0] == method_name, axis=1)]
        df = df[df.apply(lambda x: 'Total' in x.Impact_category[2], axis=1)]
    else:
        df = df[df.apply(lambda x: x.Impact_category[0] == lcia_method, axis=1)]

    return df


def normalize_lca_metrics(
        R: pd.DataFrame,
        mip_gap: float,
        lcia_method: str,
        impact_abbrev: pd.DataFrame,
        biogenic: bool = False,
        path: str = 'results/',
        metadata: dict = None,
        output: str = 'write'
) -> None | tuple[pd.DataFrame, dict]:
    """
    Create a .dat file containing the normalized LCA metrics for AMPL and a csv file containing the normalization
    factors

    :param path: path to results folder
    :param R: dataframe containing the LCA indicators results
    :param mip_gap: values lowed than the MIP gap (normalized values) are set to 0
    :param lcia_method: LCIA method to be used
    :param impact_abbrev: dataframe containing the impact categories abbreviations
    :param biogenic: whether biogenic carbon flows impact assessment method should be included or not
    :param metadata: dictionary containing the metadata. Can contain keys 'ecoinvent_version, 'year', 'spatialized',
        'regionalized', 'iam', 'ssp_rcp', 'lcia_method'.
    :param output: if 'write', writes the .dat file in 'path', if 'return', normalizes pandas dataframe, if 'both' does
        both operations.
    :return: None or the normalized pandas dataframe and the refactor dictionary (depending on the value of 'output')
    """
    if metadata is None:
        metadata = {}

    if not biogenic:
        list_biogenic_cat = ["CFB", "REQDB", "m_CCLB", "m_CCSB", "TTEQB", "TTHHB", "CCEQSB", "CCEQLB", "CCHHSB",
                             "CCHHLB", "MALB", "MASB"]
        impact_abbrev.drop(impact_abbrev[impact_abbrev.Abbrev.isin(list_biogenic_cat)].index, inplace=True)

    R = from_str_to_tuple(R, 'Impact_category')
    impact_abbrev = from_str_to_tuple(impact_abbrev, 'Impact_category')

    impact_abbrev = restrict_lcia_metrics(impact_abbrev, lcia_method)

    R = pd.merge(R, impact_abbrev, on='Impact_category')

    refactor = {}
    R_scaled = R[R['Type'] != 'Construction']
    for impact_category in R['Abbrev'].unique():
        # Scale the construction metrics to be at the same order of magnitude as the operation and resource metrics
        lcia_op_max = R[((R['Type'] == 'Operation') |
                         (R['Type'] == 'Resource')) & (R['Abbrev'] == impact_category)]['Value'].max()
        lcia_constr_max = R[(R['Type'] == 'Construction') & (R['Abbrev'] == impact_category)]['Value'].max()
        refactor[impact_category] = lcia_op_max / lcia_constr_max
        R_constr_imp = R[(R['Type'] == 'Construction') & (R['Abbrev'] == impact_category)]
        R_constr_imp['Value'] *= refactor[impact_category]
        R_scaled = pd.concat([R_scaled, R_constr_imp])  # R matrix but with refactor applied to construction metrics

    R_scaled['max_AoP'] = R_scaled.groupby('AoP')['Value'].transform('max')
    R_scaled['Value_norm'] = R_scaled['Value'] / R_scaled['max_AoP']
    R_scaled['Value_norm'] = R_scaled['Value_norm'].apply(lambda x: x if abs(x) > mip_gap else 0)

    if (output == 'write') | (output == 'both'):

        Path(path).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist

        with open(f'{path}techs_lcia.dat', 'w') as f:

            # Write metadata at the beginning of the file
            if 'ecoinvent_version' in metadata:
                f.write(f"# Ecoinvent version: {metadata['ecoinvent_version']}\n")
            if 'spatialized' in metadata:
                f.write(f"# Spatialized database: {metadata['spatialized']}\n")
            if 'regionalized' in metadata:
                f.write(f"# Regionalized database: {metadata['regionalized']}\n")
            if 'year' in metadata:
                f.write(f"# Selected year in premise: {metadata['year']}\n")
            if 'iam' in metadata:
                f.write(f"# Selected IAM in premise: {metadata['iam']}\n")
            if 'ssp_rcp' in metadata:
                f.write(f"# Selected SSP-RCP scenario in premise: {metadata['ssp_rcp']}\n")
            if 'lcia_method' in metadata:
                f.write(f"# LCIA method: {metadata['lcia_method']}\n")
            f.write("\n")

            # Set of LCA indicators
            f.write(f"set INDICATORS := {' '.join(R_scaled['Abbrev'].unique())};\n\n")

            # Declare the refactor parameters values
            f.write('# Parameters to set the operation and infrastructure indicators at the same order of magnitude\n')
            for impact_category in R_scaled['Abbrev'].unique():
                f.write(f"let refactor['{impact_category}'] := {refactor[impact_category]};\n")
            f.write('\n')

            # Declare the LCA indicators values
            for i in range(len(R_scaled)):
                # Declaring the LCIA parameters
                f.write(f"let lcia_{tech_type(R_scaled.Type.iloc[i])}['{R_scaled.Abbrev.iloc[i]}','{R_scaled.Name.iloc[i]}'] "
                        f":= {R_scaled.Value_norm.iloc[i]}; # normalized {R_scaled.Unit.iloc[i]}\n")

        # To come back to the original values, we save the maximum value of each AoP
        R_scaled[['AoP', 'max_AoP']].drop_duplicates().to_csv(f'{path}res_lcia_max.csv', index=False)

        if output == 'both':
            return R_scaled, refactor

    elif output == 'return':
        return R_scaled, refactor

    else:
        raise ValueError(f"The output parameter must be either 'write', 'return' or 'both'")
