from .utils import *
import bw2calc as bc
import pandas as pd
import ast
pd.options.mode.chained_assignment = None  # default='warn'


def get_impact_categories(methods: list[str]) -> list[str]:
    """
    Get all impact categories from a list of methods

    :param methods: list of methods
    :return: list of impact categories
    """
    return [i for i in bd.methods if i[0] in methods]


def is_empty(row: pd.Series, technology_compositions: pd.DataFrame) -> float:
    """
    Fill empty cells with the ESM value if the technology is not in the technology compositions file

    :param row: row of the lifetime dataframe
    :param technology_compositions: technology compositions dataframe
    :return: lifetime value
    """
    if (row.Name not in list(technology_compositions.Name)) & (pd.isna(row.LCA)):
        return row.ESM
    else:
        return row.LCA


def compute_impact_scores(esm_db: list[dict], mapping: pd.DataFrame, technology_compositions: pd.DataFrame,
                          methods: list[str], unit_conversion: pd.DataFrame,
                          lifetime: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute the impact scores of the technologies and resources

    :param esm_db: LCI database of the ESM
    :param mapping: mapping file
    :param technology_compositions: technology compositions file
    :param methods: list of impact assessment methods
    :param unit_conversion: unit conversion file
    :param lifetime: lifetime file
    :return: impact scores of the technologies and resources for all impact categories in the methods
    """

    esm_db_name = esm_db[0]['database']
    esm_db_dict_code = database_list_to_dict(esm_db, 'code')

    activities = [esm_db_dict_code[(esm_db_name, mapping['New_code'].iloc[i])] for i in range(len(mapping))]
    activities_bw = {(i['database'], i['code']): i for i in activities}
    impact_categories = get_impact_categories(methods)
    bd.calculation_setups['impact_scores'] = {
        'inv': [{key: 1} for key in list(activities_bw.keys())],
        'ia': impact_categories
    }
    multilca = bc.MultiLCA('impact_scores')  # computes the LCA scores

    R = pd.DataFrame(
        multilca.results,
        index=[key[1] for key in list(multilca.all.keys())],
        columns=[i for i in impact_categories]
    ).T  # save the LCA scores in a dataframe

    unit_conversion_code = pd.merge(mapping[['Name', 'Type', 'New_code']], unit_conversion, on=['Name', 'Type'],
                                    how='left')
    unit_conversion_code = pd.Series(data=unit_conversion_code.Value.values, index=unit_conversion_code.New_code)

    R = R * unit_conversion_code[R.columns]  # multiply each column by its unit conversion factor

    R_tech_op = R[list(mapping[mapping.Type == 'Operation'].New_code)]
    R_tech_constr = R[list(mapping[mapping.Type == 'Construction'].New_code)]
    R_res = R[list(mapping[mapping.Type == 'Resource'].New_code)]

    if lifetime is None:
        pass
    else:
        lifetime['LCA'] = lifetime.apply(lambda row: is_empty(row, technology_compositions), axis=1)
        lifetime_lca_code = pd.merge(mapping[mapping.Type == 'Construction'][['Name', 'New_code']], lifetime, on='Name')
        lifetime_lca_code = pd.Series(data=lifetime_lca_code.LCA.values, index=lifetime_lca_code.New_code)

        # divide each column (construction only) by its lifetime
        R_tech_constr = R_tech_constr / lifetime_lca_code[R_tech_constr.columns]

    # Reading the list of subcomponents as a list (and not as a string)
    try:
        technology_compositions.Components = technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    # Maximum length of list of subcomponents
    N_subcomp_max = max(len(i) for i in technology_compositions.Components)

    technology_compositions['Type'] = len(technology_compositions) * ['Construction']
    # Associate new code to composition of technologies
    technology_compositions['New_code'] = technology_compositions.apply(lambda row: random_code(), axis=1)

    for i in range(len(technology_compositions)):
        for j in range(len(technology_compositions.Components.iloc[i])):
            technology_compositions.loc[i, 'Component_' + str(j+1)] = technology_compositions.Components.iloc[i][j]

    # Find the new codes of the subcomponents
    for i in range(1, N_subcomp_max + 1):
        technology_compositions = pd.merge(technology_compositions,
                                           mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
                                           left_on=f'Component_{i}',
                                           right_on='Name',
                                           suffixes=('', f'_component_{i}'),
                                           how='left'
                                           ).drop(columns=f'Name_component_{i}')

    for i in range(len(technology_compositions)):
        tech_name = technology_compositions.iloc[i].Name
        subcomp_list = technology_compositions.iloc[i].Components
        new_code_composition = technology_compositions.iloc[i].New_code
        R_tech_constr[new_code_composition] = len(impact_categories) * [0]  # initialize the new column

        for j in range(1, len(subcomp_list) + 1):
            R_tech_constr[new_code_composition] += R_tech_constr[
                technology_compositions.iloc[i][f'New_code_component_{j}']
            ]  # sum up the impacts of the subcomponents

        R_tech_constr[new_code_composition] *= float(
            unit_conversion[(unit_conversion.Name == tech_name)
                            & (unit_conversion.Type == 'Construction')].Value.iloc[0]
        )  # multiply the composition column with its unit conversion factor

        R_tech_constr.drop(columns=[technology_compositions.iloc[i][f'New_code_component_{j}']
                                    for j in range(1, len(subcomp_list) + 1)], inplace=True)
        # remove subcomponents from dataframe

    if lifetime is None:
        pass
    else:
        lifetime_esm_code = pd.merge(pd.concat([mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
                                                technology_compositions[['Name', 'New_code']]]), lifetime, on='Name')
        lifetime_esm_code = pd.Series(data=lifetime_esm_code.ESM.values, index=lifetime_esm_code.New_code)
        R_tech_constr = R_tech_constr * lifetime_esm_code[R_tech_constr.columns]  # multiply by lifetime of ESM

    name_to_new_code = pd.concat([mapping[['Name', 'Type', 'New_code']],
                                 technology_compositions[['Name', 'Type', 'New_code']]])

    R_long = pd.concat([R_tech_constr, R_tech_op, R_res], axis=1).melt(ignore_index=False, var_name='New_code')
    R_long = R_long.reset_index().merge(right=name_to_new_code, on='New_code')
    R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)

    return R_long
