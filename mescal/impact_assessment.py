import bw2calc as bc
import bw2data as bd
import bw2analyzer as ba
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
from .utils import random_code
from .database import Database

pd.options.mode.chained_assignment = None  # default='warn'

try:
    from bw2data import calculation_setups
except ImportError:
    calculation_setups = None


def compute_impact_scores(
        self,
        methods: list[str],
        specific_lcia_categories: list[str] = None,
        specific_lcia_abbrev: list[str] = None,
        impact_abbrev: pd.DataFrame = None,
        assessment_type: str = 'esm',
        overwrite: bool = True,
        contribution_analysis: str = None,
        contribution_analysis_limit_type: str = 'number',
        contribution_analysis_limit: float or int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Compute the impact scores of the technologies and resources

    :param methods: list of life-cycle impact assessment methods for which LCA scores are computed
    :param specific_lcia_categories: restrict the impact assessment to specific LCIA categories
    :param specific_lcia_abbrev: restrict the impact assessment to specific LCIA categories identified by their
        abbreviation in the impact_abbrev file
    :param impact_abbrev: dataframe containing the impact categories abbreviations
    :param assessment_type: type of assessment, can be 'esm' for the computation of the energy system life-cycle
        impacts, or 'direct emissions' for the computation of direct emissions only
    :param overwrite: only relevant if assessment_type is 'direct emissions', if True, the direct emissions database
        will be overwritten if it exists
    :param contribution_analysis: if 'emissions', the function will return the contribution analysis of top elementary
        flows. If 'processes', the function will return the contribution analysis of top processes.
        If 'both', it will return both.
    :param contribution_analysis_limit_type: contribution analysis limit type, can be 'percent' or 'number'.
        Default is 'percent'.
    :param contribution_analysis_limit: number of values to return (if limit_type is 'number'), or percentage cutoff
        (if limit_type is 'percent'). Default is 0.01.
    :return: impact scores dataframe of the technologies and resources for all selected impact categories, and
        contribution analysis dataframe (would be None if contribution_analysis is None).
    """

    if assessment_type == 'direct emissions' and self.df_activities_subject_to_double_counting is None:
        self.df_activities_subject_to_double_counting = pd.read_csv(f"{self.results_path_file}activities_subject_to_double_counting.csv")

    activities_subject_to_double_counting = self.df_activities_subject_to_double_counting

    if assessment_type != 'esm' and assessment_type != 'direct emissions':
        raise ValueError('The assessment type must be either "esm" or "direct emissions"')

    if specific_lcia_categories is not None and specific_lcia_abbrev is not None:
        raise ValueError('You cannot specify both specific_lcia_categories and specific_lcia_abbrev')

    if specific_lcia_abbrev is not None and impact_abbrev is None:
        raise ValueError('You must provide the impact_abbrev dataframe if you want to use specific_lcia_abbrev')

    if contribution_analysis is not None:
        if contribution_analysis not in ['emissions', 'processes', 'both']:
            raise ValueError('The contribution_analysis must be either "emissions" or "processes"')
        if contribution_analysis_limit_type not in ['percent', 'number']:
            raise ValueError('The contribution_analysis_limit_type must be either "percent" or "number"')
        if contribution_analysis_limit_type == 'percent':
            if contribution_analysis_limit < 0 or contribution_analysis_limit > 1:
                raise ValueError('The contribution_analysis_limit must be between 0 and 1 if limit_type is "percent"')
        if contribution_analysis_limit_type == 'number':
            if contribution_analysis_limit < 0 or isinstance(contribution_analysis_limit, int) is False:
                raise ValueError('The contribution_analysis_limit must be a positive integer if limit_type is "number"')

    if self.esm_db is not None:
        esm_db = self.esm_db
    else:
        self.esm_db = Database(db_names=self.esm_db_name)
        esm_db = self.esm_db
    esm_db_dict_code = esm_db.db_as_dict_code

    if 'New_code' not in self.mapping.columns:
        self.get_new_code()

    # Store frequently accessed instance variables in local variables inside a method
    mapping = self.mapping
    esm_db_name = self.esm_db_name
    unit_conversion = self.unit_conversion
    lifetime = self.lifetime
    technology_compositions = self.technology_compositions

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
            act['exchanges'] = [exc for exc in act['exchanges'] if exc['type'] != 'technosphere']
            act['comment'] = ('Technosphere flows have been set to 0 to keep only direct emissions. '
                              + act.get('comment', ''))
            for exc in act['exchanges']:  # change database name in exchanges
                if exc['type'] == 'production':
                    exc['database'] = esm_direct_emissions_db_name
                    if 'input' in exc:
                        exc['input'] = (esm_direct_emissions_db_name, exc['input'][1])
                if 'output' in exc:
                    exc['output'] = (esm_direct_emissions_db_name, exc['output'][1])
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

    # Filtering impact categories if specific_lcia_categories or specific_lcia_abbrev is provided
    if specific_lcia_abbrev is not None:
        try:
            impact_abbrev.Impact_category = impact_abbrev.Impact_category.apply(ast.literal_eval)
        except ValueError:
            pass
        specific_lcia_categories_full = list(impact_abbrev[impact_abbrev.Abbrev.isin(specific_lcia_abbrev)].Impact_category)
        specific_lcia_categories = [i[-1] for i in specific_lcia_categories_full]

    if specific_lcia_categories is not None:
        impact_categories = [i for i in impact_categories if i[-1] in specific_lcia_categories]

    bd.calculation_setups[calculation_setup_name] = {
        'inv': [{key: 1} for key in list(activities_bw.keys())],
        'ia': impact_categories
    }

    multilca = MultiLCA(  # computes the LCA scores
        cs_name=calculation_setup_name,
        contribution_analysis=contribution_analysis,
        limit=contribution_analysis_limit,
        limit_type=contribution_analysis_limit_type
    )

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

    if contribution_analysis is not None:
        df_contrib_analysis_results = multilca.df_res_concat

        df_contrib_analysis_results = pd.merge(  # adding the Name and Type columns to the dataframe
            left=df_contrib_analysis_results,
            right=mapping[['Name', 'Type', 'New_code']],
            left_on='act_code',
            right_on='New_code',
            how='left',
        )

        df_contrib_analysis_results = pd.merge(  # adding unit conversion factors to the dataframe
            left=df_contrib_analysis_results,
            right=unit_conversion[['Name', 'Type', 'Value']],
            on=['Name', 'Type'],
            how='left',
        )
        df_contrib_analysis_results.rename(
            columns={'Name': 'act_name', 'Type': 'act_type'},
            inplace=True
        )

        # Multiply the score and amount columns by the conversion factor
        df_contrib_analysis_results['score'] = df_contrib_analysis_results['score'] * df_contrib_analysis_results[
            'Value']
        df_contrib_analysis_results['amount'] = df_contrib_analysis_results['amount'] * df_contrib_analysis_results[
            'Value']

        df_contrib_analysis_results.drop(columns=['New_code', 'Value'], inplace=True)

    else:
        df_contrib_analysis_results = None

    if assessment_type == 'direct emissions':
        R_long = R.melt(ignore_index=False, var_name='New_code').reset_index()
        R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)
        R_long = R_long.merge(
            right=mapping[mapping.Type == 'Operation'][['Name', 'New_code', 'Type']],
            on='New_code',
            how='left'
        )

        if contribution_analysis is not None:
            return R_long, df_contrib_analysis_results
        else:
            return R_long, None

    R_tech_op = R[list(mapping[mapping.Type == 'Operation'].New_code)]
    R_tech_constr = R[list(mapping[mapping.Type == 'Construction'].New_code)]
    R_res = R[list(mapping[mapping.Type == 'Resource'].New_code)]

    if contribution_analysis is not None:
        df_contrib_analysis_results_op = df_contrib_analysis_results[
            df_contrib_analysis_results['act_type'] == 'Operation']
        df_contrib_analysis_results_constr = df_contrib_analysis_results[
            df_contrib_analysis_results['act_type'] == 'Construction']
        df_contrib_analysis_results_res = df_contrib_analysis_results[
            df_contrib_analysis_results['act_type'] == 'Resource']
    else:
        df_contrib_analysis_results_op = None
        df_contrib_analysis_results_constr = None
        df_contrib_analysis_results_res = None

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

        if contribution_analysis is not None:
            df_contrib_analysis_results_constr = pd.merge(
                left=df_contrib_analysis_results_constr,
                right=lifetime[['Name', 'LCA']],
                left_on='act_name',
                right_on='Name',
                how='left'
            )

            # divide the construction score and amount columns by the technologies lifetime in LCI datasets
            df_contrib_analysis_results_constr['score'] = (df_contrib_analysis_results_constr['score'] /
                                                           df_contrib_analysis_results_constr['LCA'])
            df_contrib_analysis_results_constr['amount'] = (df_contrib_analysis_results_constr['amount'] /
                                                            df_contrib_analysis_results_constr['LCA'])
            df_contrib_analysis_results_constr.drop(columns=['Name', 'LCA'], inplace=True)

    # Reading the list of subcomponents as a list (and not as a string)
    try:
        technology_compositions.Components = technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    # Maximum length of list of subcomponents
    N_subcomp_max = max(len(i) for i in technology_compositions.Components)

    technology_compositions['Type'] = len(technology_compositions) * ['Construction']
    # Associate new code to composition of technologies (this code does not correspond to any activity in the database,
    # it is only used as an identifier for the user)
    technology_compositions['New_code'] = technology_compositions.apply(lambda row: random_code(), axis=1)

    for i in range(len(technology_compositions)):
        for j in range(len(technology_compositions.Components.iloc[i])):
            technology_compositions.loc[i, 'Component_' + str(j + 1)] = technology_compositions.Components.iloc[i][j]

    # Find the new codes of the subcomponents
    for i in range(1, N_subcomp_max + 1):
        technology_compositions = pd.merge(
            left=technology_compositions,
            right=mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
            left_on=f'Component_{i}',
            right_on='Name',
            suffixes=('', f'_component_{i}'),
            how='left'
        ).drop(columns=f'Name_component_{i}')

    df_comp_list = []  # list to store the contribution analysis results for each technology composition
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
            unit_conversion[
                (unit_conversion.Name == tech_name)
                & (unit_conversion.Type == 'Construction')].Value.iloc[0]
        )  # multiply the composition column with its unit conversion factor

        R_tech_constr.drop(columns=[technology_compositions.iloc[i][f'New_code_component_{j}']
                                    for j in range(1, len(subcomp_list) + 1)], inplace=True)
        # remove subcomponents from dataframe

        if contribution_analysis is not None:
            # sum up the contributions of the subcomponents
            df_subcomp = df_contrib_analysis_results_constr[
                df_contrib_analysis_results_constr['act_name'].isin(subcomp_list)
            ]

            df_comp = df_subcomp.groupby([
                'code', 'database', 'impact_category', 'act_database', 'act_type'
            ]).agg({
                'score': 'sum',
                'amount': 'sum',
            }).reset_index()

            df_comp['act_name'] = tech_name  # add the technology name to the dataframe
            df_comp['act_code'] = new_code_composition  # add the technology code to the dataframe

            df_comp_list.append(df_comp)  # add the contribution analysis results to the list to be concatenated later
            df_contrib_analysis_results_constr.drop(  # remove the subcomponents from the dataframe
                df_subcomp.index, inplace=True
            )

    if contribution_analysis is not None:
        df_comp_all = pd.concat(df_comp_list)  # concatenate composition results in a single dataframe

        df_comp_all = pd.merge(  # adding the unit conversion factors to the dataframe
            left=df_comp_all,
            right=unit_conversion[['Name', 'Type', 'Value']],
            left_on=['act_name', 'act_type'],
            right_on=['Name', 'Type'],
            how='left',
        )

        # Multiply the score and amount columns by the conversion factor in the composition dataframe
        df_comp_all['score'] = (df_comp_all['score'] * df_comp_all['Value'])
        df_comp_all['amount'] = (df_comp_all['amount'] * df_comp_all['Value'])

        df_comp_all.drop(columns=['Value', 'Name', 'Type'], inplace=True)

        df_contrib_analysis_results_constr = pd.concat(  # concatenate the contribution analysis results
            [df_contrib_analysis_results_constr, df_comp_all],
            ignore_index=True
        )

    if lifetime is None:
        pass
    else:
        lifetime_esm_code = pd.merge(pd.concat([
            mapping[mapping.Type == 'Construction'][['Name', 'New_code']],
            technology_compositions[['Name', 'New_code']]]), lifetime,
            on='Name'
        )
        lifetime_esm_code = pd.Series(data=lifetime_esm_code.ESM.values, index=lifetime_esm_code.New_code)
        R_tech_constr = R_tech_constr * lifetime_esm_code[R_tech_constr.columns]  # multiply by lifetime of ESM

        if contribution_analysis is not None:
            df_contrib_analysis_results_constr = pd.merge(
                left=df_contrib_analysis_results_constr,
                right=lifetime[['Name', 'ESM']],
                left_on='act_name',
                right_on='Name',
                how='left'
            )

            # multiply the construction score and amount columns by the technologies lifetime in the ESM
            df_contrib_analysis_results_constr['score'] = (df_contrib_analysis_results_constr['score'] *
                                                           df_contrib_analysis_results_constr['ESM'])
            df_contrib_analysis_results_constr['amount'] = (df_contrib_analysis_results_constr['amount'] *
                                                            df_contrib_analysis_results_constr['ESM'])

            df_contrib_analysis_results_constr.drop(columns=['Name', 'ESM'], inplace=True)

    name_to_new_code = pd.concat([mapping[['Name', 'Type', 'New_code']],
                                  technology_compositions[['Name', 'Type', 'New_code']]])

    R_long = pd.concat([R_tech_constr, R_tech_op, R_res], axis=1).melt(ignore_index=False, var_name='New_code')
    R_long = R_long.reset_index().merge(right=name_to_new_code, on='New_code')
    R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)

    if contribution_analysis is not None:
        df_contrib_analysis_results = pd.concat(
            [df_contrib_analysis_results_constr, df_contrib_analysis_results_op, df_contrib_analysis_results_res],
            ignore_index=True,
        )

        return R_long, df_contrib_analysis_results

    else:
        return R_long, None


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
                        exc['input'] = (esm_direct_emissions_db_name, exc['input'][1])
                if 'output' in exc:
                    exc['output'] = (esm_direct_emissions_db_name, exc['output'][1])

        # In the case of forced background check, we keep only the main activity, and remove the other ones
        elif tech in self.activities_background_search:
            for i in range(len(activities)):
                if activities['Activity code'].iloc[i] == old_act['code'] and activities['Amount'].iloc[i] == 1.0:
                    act = [j for j in direct_emissions_db if j['code'] == activities['Activity code'].iloc[i]][0]
                    act['name'] = f'{tech}, Operation'
                    for exc in act['exchanges']:
                        if exc['type'] == 'production':
                            exc['name'] = f'{tech}, Operation'
                            exc['database'] = esm_direct_emissions_db_name
                            if 'input' in exc:
                                exc['input'] = (esm_direct_emissions_db_name, exc['input'][1])
                        if 'output' in exc:
                            exc['output'] = (esm_direct_emissions_db_name, exc['output'][1])
                else:  # remove activity from database
                    direct_emissions_db = [j for j in direct_emissions_db if j['code'] != activities['Activity code'].iloc[i]]

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


def bw2_compat_annotated_top_emissions(lca, names=True, **kwargs):
    """
    Get list of most damaging biosphere flows in an LCA, sorted by ``abs(direct impact)``.

    Returns a list of tuples: ``(lca score, inventory amount, activity)``. If ``names`` is False, they return the
        process key as the last element.
    """
    # This is a temporary fix, until https://github.com/brightway-lca/brightway2-analyzer/issues/27

    ra, rp, rb = lca.reverse_dict()
    results = [
        (score, lca.inventory[int(index), :].sum(), rb[int(index)])
        for score, index in ba.ContributionAnalysis().top_emissions(
            lca.characterized_inventory, **kwargs
        )
    ]
    if names:
        results = [(x[0], x[1], bd.get_activity(x[2])) for x in results]
    return results


def bw2_compat_annotated_top_processes(lca, names=True, **kwargs):
    """
    Get list of most damaging processes in an LCA, sorted by ``abs(direct impact)``.

    Returns a list of tuples: ``(lca score, supply, activity)``. If ``names`` is False, they return the process
        key as the last element.
    """
    # This is a temporary fix, until https://github.com/brightway-lca/brightway2-analyzer/issues/27

    ra, rp, rb = lca.reverse_dict()
    results = [
        (score, lca.supply_array[int(index)], ra[int(index)])
        for score, index in ba.ContributionAnalysis().top_processes(
            lca.characterized_inventory, **kwargs
        )
    ]
    if names:
        results = [(x[0], x[1], bd.get_activity(x[2])) for x in results]
    return results


class MultiLCA(object):
    """
    Adaptation of the `MultiLCA` class from the `bw2calc` package in order to perform contribution analysis.

    Wrapper class for performing LCA calculations with many functional units and LCIA methods.
    Needs to be passed a ``calculation_setup`` name.
    This class does not subclass the `LCA` class, and performs all calculations upon instantiation.
    Initialization creates `self.results`, which is a NumPy array of LCA scores, with rows of functional units and
        columns of LCIA methods. Ordering is the same as in the `calculation_setup`.
    """

    def __init__(self, cs_name, contribution_analysis, limit, limit_type, log_config=None):
        """
        Initialize the MultiLCA_with_contribution_analysis class.

        :param cs_name: name of the calculation setup to use
        :param contribution_analysis: if True, the function will perform contribution analysis
        :param limit: number of values to return (if limit_type is 'number'), or percentage cutoff (if limit_type is
            'percent')
        :param limit_type: contribution analysis limit type, can be 'percent' or 'number'
        :param log_config: log configuration for the LCA calculation
        """

        if calculation_setups is None:
            raise ImportError
        assert cs_name in calculation_setups
        try:
            cs = calculation_setups[cs_name]
        except KeyError:
            raise ValueError(
                "{} is not a known `calculation_setup`.".format(cs_name)
            )

        self.contribution_analysis = contribution_analysis
        self.limit = limit
        self.limit_type = limit_type
        df_res_list = []

        self.func_units = cs['inv']
        self.methods = cs['ia']
        fu_all = self.all
        self.lca = bc.LCA(demand=fu_all, method=self.methods[0], log_config=log_config)

        self.lca.lci(factorize=True)
        self.method_matrices = []
        self.results = np.zeros((len(self.func_units), len(self.methods)))

        for method in self.methods:
            self.lca.switch_method(method)
            self.method_matrices.append(self.lca.characterization_matrix)

        for row, func_unit in tqdm(enumerate(self.func_units)):
            self.lca.redo_lci(func_unit)
            for col, cf_matrix in enumerate(self.method_matrices):
                self.lca.characterization_matrix = cf_matrix
                self.lca.lcia_calculation()
                self.results[row, col] = self.lca.score

                if contribution_analysis in ['emissions', 'both']:
                    top_contributors = bw2_compat_annotated_top_emissions(
                        self.lca,
                        limit=self.limit,
                        limit_type=self.limit_type
                    )

                    df_res = pd.DataFrame(
                        data=[[i[0], i[1], i[2].as_dict()['code'], i[2].as_dict()['database']] for i in top_contributors],
                        columns=['score', 'amount', 'code', 'database'],
                    )

                    # Drop rows where the score is zero
                    df_res.drop(df_res[df_res['score'] == 0].index, inplace=True)

                    if len(df_res) > 0:
                        df_res['impact_category'] = len(df_res) * [self.methods[col]]

                        act = list(fu_all.keys())[row]
                        df_res['act_database'] = len(df_res) * [act[0]]
                        df_res['act_code'] = len(df_res) * [act[1]]

                        df_res_list.append(df_res)

                if contribution_analysis in ['processes', 'both']:
                    top_contributors = bw2_compat_annotated_top_processes(
                        self.lca,
                        limit=self.limit,
                        limit_type=self.limit_type
                    )

                    df_res = pd.DataFrame(
                        data=[[i[0], i[1], i[2].as_dict()['code'], i[2].as_dict()['database']] for i in top_contributors],
                        columns=['score', 'amount', 'code', 'database'],
                    )

                    # Drop rows where the score is zero
                    df_res.drop(df_res[df_res['score'] == 0].index, inplace=True)

                    if len(df_res) > 0:
                        df_res['impact_category'] = len(df_res) * [self.methods[col]]

                        act = list(fu_all.keys())[row]
                        df_res['act_database'] = len(df_res) * [act[0]]
                        df_res['act_code'] = len(df_res) * [act[1]]

                        df_res_list.append(df_res)

        if contribution_analysis is not None:
            self.df_res_concat = pd.concat(df_res_list, ignore_index=True)

    @property
    def all(self):
        """Get all possible databases by merging all functional units"""
        return {key: 1 for func_unit in self.func_units for key in func_unit}
