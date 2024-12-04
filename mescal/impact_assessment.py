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
        assessment_type: str = 'esm',
        activities_subject_to_double_counting: pd.DataFrame = None,
        overwrite: bool = True,
) -> pd.DataFrame:
    """
    Compute the impact scores of the technologies and resources

    :param methods: list of life-cycle impact assessment methods for which LCA scores are computed
    :param assessment_type: type of assessment, can be 'esm' for the full LCA database, or 'direct emissions' for the
        computation of territorial emissions only
    :param activities_subject_to_double_counting: activities that were subject to the double counting removal
    :param overwrite: only relevant if assessment_type is 'direct emissions', if True, the direct emissions database
        will be overwritten if it exists
    :return: impact scores of the technologies and resources for all impact categories of all LCIA methods
    """
    if assessment_type == 'direct emissions' and activities_subject_to_double_counting is None:
        raise ValueError('For territorial emissions computation, the activities_subject_to_double_counting dataframe '
                         'must be provided')

    if assessment_type != 'esm' and assessment_type != 'direct emissions':
        raise ValueError('The assessment type must be either "esm" or "direct emissions"')

    # Store frequently accessed instance variables in local variables inside a method
    mapping = self.mapping
    esm_db_name = self.esm_db_name
    unit_conversion = self.unit_conversion
    lifetime = self.lifetime

    esm_db = Database(db_names=esm_db_name)
    esm_db_dict_code = esm_db.db_as_dict_code

    if assessment_type == 'esm':
        calculation_setup_name = 'impact_scores'
        activities = [esm_db_dict_code[(esm_db_name, new_code)]
                      for new_code in list(mapping[mapping.Type != 'Flow'].New_code)]

    elif assessment_type == 'direct emissions':
        esm_direct_emissions_db_name = esm_db_name + '_direct_emissions'
        calculation_setup_name = 'direct_emissions'
        activities_subject_to_double_counting['Type'] = 'Operation'
        activities_subject_to_double_counting['Database'] = esm_direct_emissions_db_name

        # Filtering the database to keep only the activities subject to double counting (i.e., the ones with
        # direct emissions)
        activities = [i for i in esm_db.db_as_list if
                      i['code'] in list(activities_subject_to_double_counting['Activity code'].unique())]

        # Set the amount of all technosphere exchanges to 0 (keeps direct emissions only) and change database name
        for act in activities:
            act['database'] = esm_direct_emissions_db_name  # change database name
            # remove all technosphere exchanges
            # act['exchanges'] = [exc for exc in act['exchanges'] if exc['type'] != 'technosphere']
            act['comment'] = ('Technosphere flows have been set to 0 to keep only direct emissions. '
                              + act.get('comment', ''))
            for exc in act['exchanges']:  # change database name in exchanges
                if exc['type'] == 'production':
                    exc['database'] = esm_direct_emissions_db_name
                    if 'input' in exc:
                        exc['input'][0] = esm_direct_emissions_db_name
                if 'output' in exc:
                    exc['output'][0] = esm_direct_emissions_db_name
                # if exc['type'] == 'technosphere':
                #     exc['amount'] *= 1e-10  # set technosphere exchanges to 0

        direct_emissions_db = self.aggregate_direct_emissions_activities(
            esm_db=esm_db,
            direct_emissions_db=activities,
            activities_subject_to_double_counting=activities_subject_to_double_counting,
        )  # aggregate activities subject to double counting for each ESM technology
        direct_emissions_db.write_to_brightway(new_db_name=esm_direct_emissions_db_name, overwrite=overwrite)
        direct_emissions_db_dict_code = direct_emissions_db.db_as_dict_code
        activities = [direct_emissions_db_dict_code[(esm_direct_emissions_db_name, new_code)]
                      for new_code in list(mapping[mapping.Type == 'Operation'].New_code)]

    else:
        raise ValueError('The assessment type must be either "esm" or "direct emissions')

    activities_bw = {(i['database'], i['code']): i for i in activities}
    impact_categories = self.get_impact_categories(methods)
    bd.calculation_setups[calculation_setup_name] = {
        'inv': [{key: 1} for key in list(activities_bw.keys())],
        'ia': impact_categories
    }
    multilca = bc.MultiLCA(calculation_setup_name)  # computes the LCA scores

    R = pd.DataFrame(
        multilca.results,
        index=[key[1] for key in list(multilca.all.keys())],
        columns=[i for i in impact_categories]
    ).T  # save the LCA scores in a dataframe

    unit_conversion_code = pd.merge(
        left=mapping[['Name', 'Type', 'New_code']],
        right=unit_conversion,
        on=['Name', 'Type'],
        how='left',
    )
    unit_conversion_code = pd.Series(
        data=unit_conversion_code['Value'].values,
        index=unit_conversion_code['New_code'],
    )

    R = R * unit_conversion_code[R.columns]  # multiply each column by its unit conversion factor

    if assessment_type == 'direct emissions':
        # TODO: reformat R correctly
        # R_long = R.melt(ignore_index=False, var_name='Activity code')
        # R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)
        # R_long = R_long.reset_index().merge(right=activities_subject_to_double_counting[['Name', 'Activity code']],
        #                                     on='Activity code')
        return R

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


@staticmethod
def get_impact_categories(methods: list[str]) -> list[str]:
    """
    Get all impact categories from a list of methods

    :param methods: list of LCIA methods
    :return: list of impact categories in the LCIA methods
    """
    return [i for i in bd.methods if i[0] in methods]


def aggregate_direct_emissions_activities(
        self,
        esm_db: Database,
        direct_emissions_db: list[dict],
        activities_subject_to_double_counting: pd.DataFrame,
) -> Database:
    """
    Aggregate the activities subject to double counting for the same ESM technology

    :param esm_db: ESM database
    :param direct_emissions_db: direct emissions ESM database before aggregation
    :param activities_subject_to_double_counting: dataframe of activities subject to double counting
    :return: aggregated direct emissions ESM database
    """
    esm_db_name = self.esm_db_name
    esm_direct_emissions_db_name = esm_db_name + '_direct_emissions'

    for tech in activities_subject_to_double_counting['Name'].unique():
        activities = activities_subject_to_double_counting[activities_subject_to_double_counting.Name == tech]
        old_act = [i for i in esm_db.db_as_list if i['name'] == f'{tech}, Operation'][0]

        if (
                (len(activities) == 1)
                & (activities.iloc[0]['Amount'] == 1.0)
                & (activities.iloc[0]['Activity code'] == old_act['code'])
                ):
            act = [i for i in direct_emissions_db if i['code'] == activities.iloc[0]['Activity code']][0]
            act['name'] = f'{tech}, Operation'
            for exc in act['exchanges']:
                if exc['type'] == 'production':
                    exc['name'] = f'{tech}, Operation'
                    exc['database'] = esm_direct_emissions_db_name
                    if 'input' in exc:
                        exc['input'][0] = esm_direct_emissions_db_name
                if 'output' in exc:
                    exc['output'][0] = esm_direct_emissions_db_name

        else:
            exchanges = [
                {
                    "amount": 1,
                    "name": f'{tech}, Operation',
                    "product": old_act['reference product'],
                    "location": old_act['location'],
                    "database": esm_direct_emissions_db_name,
                    "code": old_act['code'],
                    "type": "production",
                    "unit": old_act['unit'],
                }
            ]
            for i in range(len(activities)):
                exc_code = activities.iloc[i]['Activity code']
                exc_amount = activities.iloc[i]['Amount']
                exc_act = [i for i in direct_emissions_db if i['code'] == exc_code][0]
                new_exc = {
                    "name": exc_act['name'],
                    "product": exc_act['reference product'],
                    "location": exc_act['location'],
                    "unit": exc_act['unit'],
                    "amount": exc_amount,
                    "database": esm_direct_emissions_db_name,
                    "code": exc_code,
                    "type": "technosphere",
                }
                exchanges.append(new_exc)
            new_act = {
                "name": f'{tech}, Operation',
                "reference product": old_act['reference product'],
                "location": old_act['location'],
                "code": old_act['code'],
                "unit": old_act['unit'],
                "database": esm_direct_emissions_db_name,
                "exchanges": exchanges,
            }
            direct_emissions_db.append(new_act)  # add new activity

    return Database(db_as_list=direct_emissions_db)


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


def add_virtual_technosphere_flow(
        act: dict,
        exc_act: dict,
        amount: float
) -> dict:
    """
    Add a technosphere exchange to an activity

    :param act: activity to which the exchange is added
    :param exc_act: activity of the exchange
    :param amount: amount of the exchange
    :return: the activity with the new exchange
    """
    new_exc = {
        "name": exc_act['name'],
        "product": exc_act['reference product'],
        "location": exc_act['location'],
        "unit": exc_act['unit'],
        "amount": amount,
        "database": exc_act['database'],
        "code": exc_act['code'],
        "type": "technosphere",
    }
    act['exchanges'].append(new_exc)
    return act
