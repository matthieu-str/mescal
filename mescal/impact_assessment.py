import bw2calc as bc
import bw2data as bd
import pandas as pd
import ast
from .utils import random_code
from .database import Database
pd.options.mode.chained_assignment = None  # default='warn'


def compute_impact_scores(
        self,
        methods: list[str],
) -> pd.DataFrame:
    """
    Compute the impact scores of the technologies and resources

    :param methods: list of life-cycle impact assessment methods for which LCA scores are computed
    :return: impact scores of the technologies and resources for all impact categories of all LCIA methods
    """

    # Store frequently accessed instance variables in local variables inside a method
    mapping = self.mapping
    esm_db_name = self.esm_db_name
    unit_conversion = self.unit_conversion
    lifetime = self.lifetime

    esm_db = Database(db_names=esm_db_name)
    esm_db_dict_code = esm_db.db_as_dict_code

    activities = [esm_db_dict_code[(esm_db_name, mapping['New_code'].iloc[i])] for i in
                  range(len(mapping[mapping.Type != 'Flow']))]
    activities_bw = {(i['database'], i['code']): i for i in activities}
    impact_categories = self.get_impact_categories(methods)
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

    unit_conversion_code = pd.merge(
        left=mapping[['Name', 'Type', 'New_code']],
        right=unit_conversion,
        on=['Name', 'Type'],
        how='left'
    )
    unit_conversion_code = pd.Series(data=unit_conversion_code.Value.values, index=unit_conversion_code.New_code)

    R = R * unit_conversion_code[R.columns]  # multiply each column by its unit conversion factor

    R_tech_op = R[list(mapping[mapping.Type == 'Operation'].New_code)]
    R_tech_constr = R[list(mapping[mapping.Type == 'Construction'].New_code)]
    R_res = R[list(mapping[mapping.Type == 'Resource'].New_code)]

    if lifetime is None:
        pass
    else:
        lifetime['LCA'] = lifetime.apply(lambda row: self.is_empty(row), axis=1)
        lifetime_lca_code = pd.merge(
            left=mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
            right=lifetime,
            on='Name'
        )
        lifetime_lca_code = pd.Series(data=lifetime_lca_code.LCA.values, index=lifetime_lca_code.New_code)

        # divide each column (construction only) by its lifetime
        R_tech_constr = R_tech_constr / lifetime_lca_code[R_tech_constr.columns]

    # Reading the list of subcomponents as a list (and not as a string)
    try:
        self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    # Maximum length of list of subcomponents
    N_subcomp_max = max(len(i) for i in self.technology_compositions.Components)

    self.technology_compositions['Type'] = len(self.technology_compositions) * ['Construction']
    # Associate new code to composition of technologies
    self.technology_compositions['New_code'] = self.technology_compositions.apply(lambda row: random_code(), axis=1)

    for i in range(len(self.technology_compositions)):
        for j in range(len(self.technology_compositions.Components.iloc[i])):
            self.technology_compositions.loc[i, 'Component_' + str(j + 1)] = self.technology_compositions.Components.iloc[i][j]

    # Find the new codes of the subcomponents
    for i in range(1, N_subcomp_max + 1):
        self.technology_compositions = pd.merge(
            left=self.technology_compositions,
            right=mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
            left_on=f'Component_{i}',
            right_on='Name',
            suffixes=('', f'_component_{i}'),
            how='left'
        ).drop(columns=f'Name_component_{i}')

    for i in range(len(self.technology_compositions)):
        tech_name = self.technology_compositions.iloc[i].Name
        subcomp_list = self.technology_compositions.iloc[i].Components
        new_code_composition = self.technology_compositions.iloc[i].New_code
        R_tech_constr[new_code_composition] = len(impact_categories) * [0]  # initialize the new column

        for j in range(1, len(subcomp_list) + 1):
            R_tech_constr[new_code_composition] += R_tech_constr[
                self.technology_compositions.iloc[i][f'New_code_component_{j}']
            ]  # sum up the impacts of the subcomponents

        R_tech_constr[new_code_composition] *= float(
            unit_conversion[
                (unit_conversion.Name == tech_name)
                & (unit_conversion.Type == 'Construction')].Value.iloc[0]
        )  # multiply the composition column with its unit conversion factor

        R_tech_constr.drop(columns=[self.technology_compositions.iloc[i][f'New_code_component_{j}']
                                    for j in range(1, len(subcomp_list) + 1)], inplace=True)
        # remove subcomponents from dataframe

    if lifetime is None:
        pass
    else:
        lifetime_esm_code = pd.merge(pd.concat([
            mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
            self.technology_compositions[['Name', 'New_code']]]), lifetime,
            on='Name'
        )
        lifetime_esm_code = pd.Series(data=lifetime_esm_code.ESM.values, index=lifetime_esm_code.New_code)
        R_tech_constr = R_tech_constr * lifetime_esm_code[R_tech_constr.columns]  # multiply by lifetime of ESM

    name_to_new_code = pd.concat([mapping[['Name', 'Type', 'New_code']],
                                  self.technology_compositions[['Name', 'Type', 'New_code']]])

    R_long = pd.concat([R_tech_constr, R_tech_op, R_res], axis=1).melt(ignore_index=False, var_name='New_code')
    R_long = R_long.reset_index().merge(right=name_to_new_code, on='New_code')
    R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)

    return R_long


def get_impact_categories(methods: list[str]) -> list[str]:
    """
    Get all impact categories from a list of methods

    :param methods: list of LCIA methods
    :return: list of impact categories in the LCIA methods
    """
    return [i for i in bd.methods if i[0] in methods]


def is_empty(self, row: pd.Series) -> float:
    """
    Fill empty cells with the ESM value if the technology is not in the technology compositions file

    :param row: row of the lifetime dataframe
    :return: lifetime value
    """
    if (row.Name not in list(self.technology_compositions.Name)) & (pd.isna(row.LCA)):
        return row.ESM
    else:
        return row.LCA
