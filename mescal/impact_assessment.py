import bw2calc as bc
import bw2data as bd
import bw2analyzer as ba
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
from .utils import random_code, expand_impact_category_levels
from .database import Database
from .contribution_analysis import ABContributionAnalysis

ca = ABContributionAnalysis()

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
        req_technosphere: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
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
    :param req_technosphere: if True, the function will compute the requirements for technosphere flows.
    :return: impact scores dataframe of the technologies and resources for all selected impact categories,
        contribution analysis dataframe (None if contribution_analysis is None), and technosphere flows requirements
        dataframe (None if req_technosphere is False).
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
            raise ValueError('The contribution_analysis must be either "emissions", "processes" or "both"')
        if contribution_analysis_limit_type not in ['percent', 'number']:
            raise ValueError('The contribution_analysis_limit_type must be either "percent" or "number"')
        if contribution_analysis_limit_type == 'percent':
            if contribution_analysis_limit < 0 or contribution_analysis_limit > 1:
                raise ValueError('The contribution_analysis_limit must be between 0 and 1 if limit_type is "percent"')
        if contribution_analysis_limit_type == 'number':
            if contribution_analysis_limit < 0 or isinstance(contribution_analysis_limit, int) is False:
                raise ValueError('The contribution_analysis_limit must be a positive integer if limit_type is "number"')

    # The ESM database is reloaded anyway in case some modifications were made via brightway
    # (e.g., using tech_specifics), thus possibly not accounted in the existing wurst database
    self.esm_db = Database(db_names=self.esm_db_name)
    esm_db = self.esm_db
    esm_db_dict_code = esm_db.db_as_dict_code

    if 'New_code' not in self.mapping.columns:
        self._get_new_code()

    if 'Current_code' not in self.mapping.columns:
        self._get_original_code()

    if self.extract_eol_from_construction is True and self.added_decom_to_input_data is False:
        self._add_decommission_datasets(add_decom_ds_to_db=False)

    # Store frequently accessed instance variables in local variables inside a method
    mapping = self.mapping
    mapping_infra = self.mapping_infra
    mapping_res = self.mapping_res
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

        # Filtering the database to keep only the activities subject to double counting (i.e., the ones with
        # direct emissions)
        activities = [
            i for i in esm_db.db_as_list if
            i['code'] in list(activities_subject_to_double_counting[activities_subject_to_double_counting.Type == 'Operation']['Activity code'].unique())
        ]

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

        direct_emissions_db = self._aggregate_direct_emissions_activities(
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

    impact_categories = self._get_impact_categories(methods)
    if len(impact_categories) == 0:
        raise ValueError('The selected impact methods are missing in your brightway project')

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
        limit_type=contribution_analysis_limit_type,
        req_technosphere=req_technosphere,
    )

    R = pd.DataFrame(
        multilca.results,
        index=[key[1] for key in list(multilca.all.keys())],
        columns=[i for i in impact_categories]
    ).T  # save the LCA scores in a dataframe

    unit_conversion = unit_conversion.merge(
        mapping[['Name', 'Type', 'New_code']],
        on=['Name', 'Type'],
        how='left',
    )

    unit_conversion_code = unit_conversion.dropna(subset=['New_code'])
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
        df_contrib_analysis_results['score'] = df_contrib_analysis_results['score'] * df_contrib_analysis_results['Value']
        df_contrib_analysis_results['amount'] = df_contrib_analysis_results['amount'] * df_contrib_analysis_results['Value']

        df_contrib_analysis_results.drop(columns=['New_code', 'Value'], inplace=True)

    else:
        df_contrib_analysis_results = None

    if req_technosphere:
        df_req_technosphere = multilca.df_req_technosphere
        # multiply each column by its unit conversion factor
        df_req_technosphere = df_req_technosphere * unit_conversion_code[df_req_technosphere.columns]
    else:
        df_req_technosphere = None

    if assessment_type == 'direct emissions':
        R_long = R.melt(ignore_index=False, var_name='New_code').reset_index()
        R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)
        R_long = R_long.merge(
            right=mapping[mapping.Type == 'Operation'][['Name', 'New_code', 'Type']],
            on='New_code',
            how='left'
        )
        R_long = expand_impact_category_levels(R_long)
        R_long['Impact_category_unit'] = R_long['Impact_category'].apply(lambda row: bd.Method(row).metadata['unit'])
        R_long = R_long.merge(
            unit_conversion[['Name', 'Type', 'ESM']],
            on=['Name', 'Type'],
            how='left',
        ).rename(columns={'ESM': 'Functional unit'})

        return R_long, df_contrib_analysis_results, None

    R_tech_op = R[list(mapping[mapping.Type == 'Operation'].New_code)]
    R_tech_constr = R[list(mapping_infra.New_code)]
    R_res = R[list(mapping_res.New_code)]

    if contribution_analysis is not None:
        df_contrib_analysis_results_op = df_contrib_analysis_results[
            df_contrib_analysis_results['act_type'] == 'Operation']
        df_contrib_analysis_results_constr = df_contrib_analysis_results[
            (df_contrib_analysis_results['act_type'] == 'Construction')
            | (df_contrib_analysis_results['act_type'] == 'Decommission')
        ]
        df_contrib_analysis_results_res = df_contrib_analysis_results[
            df_contrib_analysis_results['act_type'] == 'Resource']
    else:
        df_contrib_analysis_results_op = None
        df_contrib_analysis_results_constr = None
        df_contrib_analysis_results_res = None

    if req_technosphere:
        df_req_technosphere_op = df_req_technosphere[list(mapping[mapping.Type == 'Operation'].New_code)]
        df_req_technosphere_constr = df_req_technosphere[list(mapping_infra.New_code)]
        df_req_technosphere_res = df_req_technosphere[list(mapping_res.New_code)]
    else:
        df_req_technosphere_op = None
        df_req_technosphere_constr = None
        df_req_technosphere_res = None

    if lifetime is None:
        pass
    else:
        lifetime['LCA'] = lifetime.apply(lambda row: self._is_empty(row), axis=1)
        lifetime_lca_code = pd.merge(
            left=mapping_infra[['Name', 'New_code']],
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

        if req_technosphere:
            df_req_technosphere_constr = df_req_technosphere_constr / lifetime_lca_code[df_req_technosphere_constr.columns]

    # Reading the list of subcomponents as a list (and not as a string)
    try:
        technology_compositions.Components = technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    # Maximum length of list of subcomponents
    N_subcomp_max = max(len(i) for i in technology_compositions.Components)

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
            right=mapping_infra[['Name', 'Type', 'New_code']],
            left_on=[f'Component_{i}', 'Type'],
            right_on=['Name', 'Type'],
            suffixes=('', f'_component_{i}'),
            how='left'
        ).drop(columns=f'Name_component_{i}')

    df_comp_list = []  # list to store the contribution analysis results for each technology composition
    for i in range(len(technology_compositions)):
        tech_name = technology_compositions.iloc[i].Name
        tech_type = technology_compositions.iloc[i].Type
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
                & (unit_conversion.Type == tech_type)].Value.iloc[0]
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

        if req_technosphere:
            df_req_technosphere_constr[new_code_composition] = len(df_req_technosphere_constr) * [0]  # initialize the new column

            for j in range(1, len(subcomp_list) + 1):
                df_req_technosphere_constr[new_code_composition] += df_req_technosphere_constr[
                    technology_compositions.iloc[i][f'New_code_component_{j}']
                ]  # sum up the requirements of the subcomponents

            df_req_technosphere_constr[new_code_composition] *= float(
                unit_conversion[
                    (unit_conversion.Name == tech_name)
                    & (unit_conversion.Type == tech_type)].Value.iloc[0]
            )  # multiply the composition column with its unit conversion factor

            df_req_technosphere_constr.drop(columns=[technology_compositions.iloc[i][f'New_code_component_{j}']
                                        for j in range(1, len(subcomp_list) + 1)], inplace=True)

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
            mapping_infra[['Name', 'New_code']],
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

        if req_technosphere:
            df_req_technosphere_constr = df_req_technosphere_constr * lifetime_esm_code[df_req_technosphere_constr.columns]

    name_to_new_code = pd.concat([mapping[['Name', 'Type', 'New_code']],
                                  technology_compositions[['Name', 'Type', 'New_code']]])

    R_long = pd.concat([R_tech_constr, R_tech_op, R_res], axis=1).melt(ignore_index=False, var_name='New_code')
    R_long = R_long.reset_index().merge(right=name_to_new_code, on='New_code')
    R_long.rename(columns={'index': 'Impact_category', 'value': 'Value'}, inplace=True)
    R_long = expand_impact_category_levels(R_long)
    R_long['Impact_category_unit'] = R_long['Impact_category'].apply(lambda row: bd.Method(row).metadata['unit'])
    R_long = R_long.merge(
        unit_conversion[['Name', 'Type', 'ESM']],
        on=['Name', 'Type'],
        how='left',
    ).rename(columns={'ESM': 'Functional unit'})

    if req_technosphere:
        df_req_technosphere = pd.concat([
            df_req_technosphere_constr,
            df_req_technosphere_op,
            df_req_technosphere_res,
        ], axis=1).melt(ignore_index=False, var_name='New_code')
        df_req_technosphere.drop(index=df_req_technosphere.index[df_req_technosphere['value'] == 0], inplace=True)
        df_req_technosphere = df_req_technosphere.reset_index().merge(right=name_to_new_code, on='New_code')
        df_req_technosphere.rename(columns={
            'level_0': 'Technosphere flow database',
            'level_1': 'Technosphere flow code',
            'value': 'Amount'
        }, inplace=True)
        df_req_technosphere.drop(columns=['New_code'], inplace=True)

    if contribution_analysis is not None:
        df_contrib_analysis_results = pd.concat(
            [df_contrib_analysis_results_constr, df_contrib_analysis_results_op, df_contrib_analysis_results_res],
            ignore_index=True,
        )

    return R_long, df_contrib_analysis_results, df_req_technosphere

def validation_direct_carbon_emissions(
        self,
        R_direct: pd.DataFrame,
        lcia_method_carbon_emissions: str,
        carbon_flow_in_esm: str or list[str],
        esm_results: pd.DataFrame = None,
        return_df: bool = False,
        save_df: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None] or None:

    """
    Returns a dataframe comparing the direct carbon emissions obtained from the LCIA phase (direct emissions module)
    and direct carbon emissions from the ESM. Please make sure that carbon emissions are expressed with the same
    physical unit, e.g., kg CO2-eq., in the ESM and LCIA method (while functional units, e.g., kWh, are automatically
    converted to ESM units).

    :param R_direct: dataframe containing the direct carbon emissions from the LCIA phase
    :param lcia_method_carbon_emissions: name of the LCIA method for carbon emissions in brightway
    :param carbon_flow_in_esm: names(s) of the carbon flow(s) in the ESM
    :param esm_results: dataframe containing the annual production of each technology in the ESM. It must contain the
        columns 'Name' and 'Production', and it can possibly contain the 'Run' and 'Year' columns too. If provided, the
        system's direct emissions will be compared.
    :param return_df: if True, the function will return the dataframe
    :param save_df: if True, the function will save the dataframe to a csv file named direct_carbon_emissions_differences
         and a file named direct_carbon_emissions_differences_system (if esm_results is provided) in self.results_path_file
    :return: dataframe comparing the direct carbon emissions from the LCIA phase and the ESM if return_df is True
    """

    model = self.model
    unit_conversion = self.unit_conversion

    if save_df is False and return_df is False:
        raise ValueError('You must set at least one of the parameters save_df or return_df to True')

    if isinstance(R_direct.Impact_category.iloc[0], str):
        R_direct['Impact_category'] = R_direct['Impact_category'].apply(ast.literal_eval)

    if lcia_method_carbon_emissions not in list(R_direct['Impact_category'].unique()):
        raise ValueError(f'The LCIA method {lcia_method_carbon_emissions} is not in the impact scores dataframe')

    if isinstance(carbon_flow_in_esm, str):
        carbon_flow_in_esm = [carbon_flow_in_esm]

    R_direct = R_direct[R_direct['Impact_category'] == lcia_method_carbon_emissions]
    R_direct = R_direct[R_direct['Type'] == 'Operation']
    R_direct.rename(columns={'Value': 'LCA direct carbon emissions (LCA unit)'}, inplace=True)
    R_direct.drop(columns=['Impact_category', 'Type'], inplace=True)

    R_direct['ESM direct carbon emissions (ESM unit)'] = R_direct.apply(
        lambda row: sum(model[(model.Name == row.Name) & (model.Flow.isin(carbon_flow_in_esm))]['Amount']),
        axis=1
    )

    # Get the unit conversion factor of the output unit
    R_direct = R_direct.merge(
        unit_conversion[unit_conversion.Type == 'Operation'][['Name', 'Value', 'LCA', 'ESM']],
        how='left',
        on='Name',
    )

    R_direct.rename(
        columns={'Value': 'Output conversion factor', 'LCA': 'LCA output unit', 'ESM': 'ESM output unit'},
        inplace=True
    )

    R_direct['LCA direct carbon emissions (ESM unit)'] = (
            R_direct['LCA direct carbon emissions (LCA unit)'] * R_direct['Output conversion factor'])

    R_direct['Direct carbon emissions difference'] = (
            R_direct['ESM direct carbon emissions (ESM unit)'] - R_direct['LCA direct carbon emissions (ESM unit)'])
    R_direct['Direct carbon emissions difference (%)'] = R_direct.apply(
        lambda row: (row['Direct carbon emissions difference'] / row['LCA direct carbon emissions (ESM unit)']) * 100
        if row['LCA direct carbon emissions (ESM unit)'] != 0 else None,
        axis=1
    )

    df_columns = [
        'Name',
        'ESM direct carbon emissions (ESM unit)',
        'LCA direct carbon emissions (ESM unit)',
        'Direct carbon emissions difference',
        'Direct carbon emissions difference (%)',
        'LCA direct carbon emissions (LCA unit)',
        'LCA output unit',
        'ESM output unit',
        'Output conversion factor',
    ]

    if 'Year' in R_direct.columns:
        df_columns.insert(0, 'Year')

    R_direct = R_direct[df_columns]
    
    if esm_results is not None:

        id_columns = ['Name']
        group_by_columns = ['Run']

        if 'Year' in R_direct.columns and 'Year' in esm_results.columns:
            id_columns.append('Year')
            group_by_columns.append('Year')

        if 'Run' not in esm_results.columns:
            esm_results['Run'] = 'Total'

        R_direct_tot = R_direct.merge(esm_results, on=id_columns)
        R_direct_tot['ESM direct carbon emissions (ESM unit)'] *= R_direct_tot['Production']
        R_direct_tot['LCA direct carbon emissions (ESM unit)'] *= R_direct_tot['Production']
        R_direct_tot_grouped = R_direct_tot.groupby(group_by_columns).sum().reset_index()
        R_direct_tot_grouped['Name'] = 'Total'

        R_direct_tot = pd.concat([R_direct_tot, R_direct_tot_grouped])[
            group_by_columns + ['Name', 'Production', 'ESM direct carbon emissions (ESM unit)', 'LCA direct carbon emissions (ESM unit)']
        ]
        R_direct_tot['Direct carbon emissions difference'] = (
                R_direct_tot['ESM direct carbon emissions (ESM unit)'] - R_direct_tot['LCA direct carbon emissions (ESM unit)'])
        R_direct_tot['Direct carbon emissions difference (%)'] = R_direct_tot.apply(
            lambda row: (row['Direct carbon emissions difference'] / row['LCA direct carbon emissions (ESM unit)']) * 100
            if row['LCA direct carbon emissions (ESM unit)'] != 0 else None,
            axis=1
        )

        if save_df:
            R_direct_tot.to_csv(f"{self.results_path_file}direct_carbon_emissions_differences_system.csv", index=False)

    if save_df:
        R_direct.to_csv(f"{self.results_path_file}direct_carbon_emissions_differences.csv", index=False)

    if return_df:
        if esm_results is not None:
            return R_direct, R_direct_tot
        else:
            return R_direct, None

@staticmethod
def _get_impact_categories(methods: list[str]) -> list[str]:
    """
    Get all impact categories from a list of methods

    :param methods: list of LCIA methods
    :return: list of impact categories in the LCIA methods
    """
    all_cat = []
    for method in methods:
        cat = [i for i in bd.methods if i[0] == method]
        if len(cat) == 0:
            raise ValueError(f'The LCIA method {method} is not available in your brightway project')
        all_cat.extend(cat)

    return all_cat


def _aggregate_direct_emissions_activities(
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
    tech_list = activities_subject_to_double_counting[activities_subject_to_double_counting.Type == 'Operation']['Name'].unique()

    for tech in tech_list:
        activities = activities_subject_to_double_counting[
            (activities_subject_to_double_counting.Name == tech)
            & (activities_subject_to_double_counting.Type == 'Operation')
        ]
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
        elif tech in self.activities_background_search['Operation']:
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


def _is_empty(self, row: pd.Series) -> float:
    """
    Fill empty cells with the ESM value if the technology is not in the technology compositions file

    :param row: row of the lifetime dataframe
    :return: lifetime value
    """
    if (row.Name not in list(self.technology_compositions.Name)) & (pd.isna(row.LCA)):
        return row.ESM
    else:
        return row.LCA


def _add_virtual_technosphere_flow(
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


class MultiLCA(object):
    """
    Adaptation of the `MultiLCA` class from the `bw2calc` package in order to perform contribution analysis.

    Wrapper class for performing LCA calculations with many functional units and LCIA methods.
    Needs to be passed a ``calculation_setup`` name.
    This class does not subclass the `LCA` class, and performs all calculations upon instantiation.
    Initialization creates `self.results`, which is a NumPy array of LCA scores, with rows of functional units and
    columns of LCIA methods. Ordering is the same as in the `calculation_setup`.
    """

    def __init__(
            self,
            cs_name: str,
            contribution_analysis: str,
            limit: int or float,
            limit_type: str,
            req_technosphere: bool,
            log_config=None,
    ):
        """
        Initialize the MultiLCA_with_contribution_analysis class.

        :param cs_name: name of the calculation setup to use
        :param contribution_analysis: if 'emissions', the function will return the contribution analysis of top
            elementary flows. If 'processes', the function will return the contribution analysis of top processes.
            If 'both', it will return both.
        :param limit: number of values to return (if limit_type is 'number'), or percentage cutoff (if limit_type is
            'percent')
        :param limit_type: contribution analysis limit type, can be 'percent' or 'number'
        :param req_technosphere: if True, the function will compute the requirements for technosphere flows
        :param log_config: log configuration for the LCA calculation
        """

        if calculation_setups is None:
            raise ImportError
        assert cs_name in calculation_setups
        try:
            cs = calculation_setups[cs_name]
        except KeyError:
            raise ValueError(f"{cs_name} is not a known calculation setup")

        self.contribution_analysis = contribution_analysis
        self.limit = limit
        self.limit_type = limit_type
        df_res_rows = []
        req_tech_list = []

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

        ### LOCAL references for speed inside loop (avoid attribute lookups)
        lca = self.lca
        reverse_dict = lca.reverse_dict()
        ra, rp, rb = reverse_dict
        methods = self.methods
        method_matrices = self.method_matrices

        for row, func_unit in tqdm(enumerate(self.func_units)):

            lca.redo_lci(func_unit)

            if req_technosphere:
                req_tech = pd.Series(np.multiply(lca.supply_array, lca.technosphere_matrix.diagonal()), lca.product_dict)
                req_tech.name = list(func_unit.keys())[0][1]  # use the activity code as the column name (ESM database always)
                req_tech_list.append(req_tech)

            for col, cf_matrix in enumerate(method_matrices):
                lca.characterization_matrix = cf_matrix
                lca.lcia_calculation()
                self.results[row, col] = lca.score

                if contribution_analysis in ['emissions', 'both']:
                    flow_scores = np.asarray(lca.characterized_inventory.sum(axis=1)).ravel()
                    sorted_flows = ca.sort_array(flow_scores, limit=limit, limit_type=limit_type)

                    act = list(fu_all.keys())[row]
                    for value, idx in sorted_flows:
                        if value == 0:
                            continue
                        flow = bd.get_activity(rb[int(idx)])  # biosphere dict
                        df_res_rows.append([
                            value,
                            flow_scores[int(idx)],
                            flow['code'],
                            flow['database'],
                            methods[col],
                            act[0],  # act_database
                            act[1],  # act_code
                            "emissions", # contribution_type
                        ])

                if contribution_analysis in ['processes', 'both']:
                    process_scores = np.asarray(lca.characterized_inventory.sum(axis=0)).ravel()
                    sorted_processes = ca.sort_array(process_scores, limit=limit, limit_type=limit_type)

                    act = list(fu_all.keys())[row]
                    for value, idx in sorted_processes:
                        if value == 0:
                            continue
                        proc = bd.get_activity(ra[int(idx)])  # activity dict
                        df_res_rows.append([
                            value,
                            process_scores[int(idx)],
                            proc['code'],
                            proc['database'],
                            methods[col],
                            act[0],  # act_database
                            act[1],  # act_code
                            "processes", # contribution_type
                        ])

        if contribution_analysis is not None:
            self.df_res_concat = pd.DataFrame(
                df_res_rows,
                columns=[
                    "score",
                    "amount",
                    "code",
                    "database",
                    "impact_category",
                    "act_database",
                    "act_code",
                    "contribution_type",
                ],
            )

        if req_technosphere:
            self.df_req_technosphere = pd.concat(req_tech_list, axis=1).fillna(0)

    @property
    def all(self):
        """Get all possible databases by merging all functional units"""
        return {key: 1 for func_unit in self.func_units for key in func_unit}
