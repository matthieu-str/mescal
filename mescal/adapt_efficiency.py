import pandas as pd
import ast
from .database import Dataset


def correct_esm_and_lca_efficiency_differences(
        self,
        write_efficiency_report: bool = True,
        db_type: str = 'esm',
) -> None:
    """
    Correct the efficiency differences between ESM technologies and their operation LCI datasets. This method can be
    used during the creation of the ESM database and during the creation of the ESM results database.

    :param write_efficiency_report: if True, write the efficiency differences in a csv file
    :param db_type: type of database to use for the efficiency correction, can be either 'esm' or 'esm results'
    :return: None
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    db_dict_code = self.main_database.db_as_dict_code
    db_dict_name = self.main_database.db_as_dict_name
    mapping = self.mapping
    efficiency = self.efficiency
    unit_conversion = self.unit_conversion
    mapping_esm_flows_to_CPC_cat = self.mapping_esm_flows_to_CPC_cat
    removed_flows = self.df_flows_set_to_zero
    double_counting_removal_amount = self.double_counting_removal_amount
    esm_results_db_name = self.esm_results_db_name

    try:
        efficiency.Flow = efficiency.Flow.apply(ast.literal_eval)
    except ValueError:
        pass

    efficiency['ESM efficiency'] = efficiency.apply(
        self.compute_efficiency_esm,
        axis=1,
    )
    efficiency['LCA input unit'] = efficiency.apply(
        self.get_lca_input_flow_unit_or_product,
        axis=1,
        output_type='unit',
        removed_flows=removed_flows
    )
    efficiency.drop(efficiency[efficiency['LCA input unit'].isnull()].index, inplace=True)
    efficiency['LCA input product'] = efficiency.apply(
        self.get_lca_input_flow_unit_or_product,
        axis=1,
        output_type='product',
        removed_flows=removed_flows
    )
    efficiency.drop(efficiency[efficiency['LCA input product'].isnull()].index, inplace=True)
    efficiency['LCA input quantity'] = efficiency.apply(
        self.get_lca_input_quantity,
        axis=1,
        double_counting_removal_amount=double_counting_removal_amount
    )
    efficiency.drop(efficiency[efficiency['LCA input quantity'] == 0].index, inplace=True)
    efficiency = efficiency.merge(
        right=unit_conversion[unit_conversion.Type == 'Operation'][['Name', 'Value', 'LCA', 'ESM']],
        how='left',
        on='Name'
    )
    efficiency.rename(
        columns={'Value': 'Output conversion factor', 'LCA': 'LCA output unit', 'ESM': 'ESM output unit'},
        inplace=True)
    efficiency = efficiency.merge(
        right=unit_conversion[
            (unit_conversion.Type == 'Other')
            & (unit_conversion.ESM == 'kilowatt hour')
            ][['Name', 'Value', 'LCA']],
        how='left',
        left_on=['LCA input product', 'LCA input unit'],
        right_on=['Name', 'LCA'],
        suffixes=('', '_to_remove')
    )
    efficiency.drop(columns=['Name_to_remove', 'LCA'], inplace=True)
    efficiency.rename(columns={'Value': 'Input conversion factor'}, inplace=True)

    missing_units = efficiency[efficiency['Input conversion factor'].isna()][
        ['LCA input product', 'LCA input unit']].values.tolist()
    missing_units = [tuple(x) + ('kilowatt hour',) for x in set(tuple(x) for x in missing_units)]
    if len(missing_units) > 0:
        raise ValueError(f'No conversion factor found for the following units (product, unit from, unit to): '
                         f'{missing_units}')

    efficiency['LCA efficiency'] = efficiency['Input conversion factor'] / (
            efficiency['Output conversion factor'] * efficiency['LCA input quantity'])

    efficiency['efficiency_ratio'] = efficiency['LCA efficiency'] / efficiency['ESM efficiency']

    for i in range(len(efficiency)):

        act_to_adapt_list = []  # there might be several activities to adapt for one technology in case of market
        techno_flows_to_correct_dict = {}
        tech = efficiency['Name'].iloc[i]
        flows_list = efficiency['Flow'].iloc[i]

        CPC_list = []  # list of CPC categories corresponding to the fuel flow(s) of the technology
        for flow in flows_list:
            CPC_list += mapping_esm_flows_to_CPC_cat[mapping_esm_flows_to_CPC_cat['Flow'] == flow]['CPC'].values[0]

        df_removed_flows = removed_flows[removed_flows.Name == tech]  # flows removed during double counting removal
        for j in range(len(df_removed_flows)):
            (main_act_database, main_act_code, removed_act_database, removed_act_code) = df_removed_flows[
                ['Database', 'Code', 'Removed flow database', 'Removed flow code']].iloc[j]
            act_exc = db_dict_code[removed_act_database, removed_act_code]
            if 'classifications' in act_exc:
                if 'CPC' in dict(act_exc['classifications']):
                    if dict(act_exc['classifications'])['CPC'] in CPC_list:
                        # if this flow (that was removed during double counting removal) is a fuel flow of the
                        # technology, the biosphere flows the activity will be adjusted
                        if db_type == 'esm':
                            act_to_adapt = db_dict_code[main_act_database, main_act_code]
                        elif db_type == 'esm results':
                            act_in_esm_db = db_dict_code[main_act_database, main_act_code]
                            if act_in_esm_db['name'] == f'{tech}, Operation':
                                act_in_esm_db_name = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')]['Activity'].iloc[0]
                                if (
                                    f"{act_in_esm_db_name} ({tech})",
                                    act_in_esm_db['reference product'],
                                    act_in_esm_db['location'],
                                    esm_results_db_name,
                                ) not in db_dict_name.keys():
                                    # If the technology is not in the ESM results database, it means that its annual
                                    # production was null. Thus, we do not need to correct its efficiency.
                                    act_to_adapt = None
                                else:
                                    act_to_adapt = db_dict_name[(
                                        f"{act_in_esm_db_name} ({tech})",
                                        act_in_esm_db['reference product'],
                                        act_in_esm_db['location'],
                                        esm_results_db_name,
                                    )]
                            else:
                                if (
                                    act_in_esm_db['name'],
                                    act_in_esm_db['reference product'],
                                    act_in_esm_db['location'],
                                    esm_results_db_name,
                                ) not in db_dict_name.keys():
                                    # If the technology is not in the ESM results database, it means that its annual
                                    # production was null. Thus, we do not need to correct its efficiency.
                                    act_to_adapt = None
                                else:
                                    act_to_adapt = db_dict_name[(
                                        act_in_esm_db['name'],
                                        act_in_esm_db['reference product'],
                                        act_in_esm_db['location'],
                                        esm_results_db_name,
                                    )]
                        else:
                            raise ValueError(f'db_type must be either "esm" or "esm results"')

                        if act_to_adapt not in act_to_adapt_list and act_to_adapt is not None:
                            # in case there are several fuel flows in the same activity
                            act_to_adapt_list.append(act_to_adapt)
                            techno_flows_to_correct_dict[
                                (act_to_adapt['database'], act_to_adapt['code'])
                            ] = []

                        if act_to_adapt is not None:
                            techno_flows_to_correct_dict[
                                (act_to_adapt['database'], act_to_adapt['code'])
                            ] += [(act_exc['database'], act_exc['code'])]

        if len(act_to_adapt_list) == 0:
            if db_type == 'esm':
                self.logger.warning(
                    f'No flow of type(s) {flows_list} found for {tech}. The efficiency of this technology cannot be '
                    f'adjusted.'
                )

        for act in act_to_adapt_list:
            efficiency_ratio = efficiency['efficiency_ratio'].iloc[i]
            techno_flows_to_correct = techno_flows_to_correct_dict[(act['database'], act['code'])]
            act = self.adapt_flows_to_efficiency_difference(act, efficiency_ratio, techno_flows_to_correct)

    if write_efficiency_report:
        # saving the efficiency differences in a csv file
        efficiency.to_csv(f'{self.results_path_file}efficiency_differences.csv', index=False)


def compute_efficiency_esm(
        self,
        row: pd.Series,
) -> float:
    """
    Retrieve a technology efficiency from the ESM model file

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :return: efficiency of the technology with respect to the given flow
    """
    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    model = self.model

    flows_list = row.Flow
    input_amount = 0

    for flow in flows_list:
        try:
            input_amount += model[(model.Name == row.Name) & (model.Flow == flow)].Amount.values[0]
        except IndexError:
            # This allows the user to put a fuel that is not an input in the ESM but in the LCI dataset in the
            # efficiency file. This is useful in case of mismatch (different fuel consumed between ESM nd LCI dataset).
            # However, direct emissions must then be adjusted (tech_specifics file).
            self.logger.warning(f'Flow of type {flow} found for {row.Name} in efficiency file, but not in model file.')
            input_amount += 0
    if input_amount == 0:
        raise ValueError(f'No flow of type(s) {flows_list} found for {row.Name} in the model file')
    return -1.0 / input_amount  # had negative sign as it is an input


def get_lca_input_flow_unit_or_product(
        self,
        row: pd.Series,
        output_type: str,
        removed_flows: pd.DataFrame
) -> str | None:
    """
    Retrieve the unit or product of the removed flow (taken from the double counting removal report)

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param output_type: can be either 'unit' or 'product'
    :param removed_flows: dataframe containing the name and amount of removed flows during double counting removal
    :return: the unit or product of the flow removed for the ESM (technology, flow) couple
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    db_dict_code = self.main_database.db_as_dict_code
    mapping_esm_flows_to_CPC_cat = self.mapping_esm_flows_to_CPC_cat

    unit_list = []
    name_list = []
    CPC_list = []

    flows_list = row.Flow

    for flow in flows_list:
        CPC_list += mapping_esm_flows_to_CPC_cat[mapping_esm_flows_to_CPC_cat['Flow'] == flow]['CPC'].values[0]
    df_removed_flows = removed_flows[removed_flows.Name == row.Name]

    for i in range(len(df_removed_flows)):
        removed_act_database, removed_act_code = df_removed_flows[
            ['Removed flow database', 'Removed flow code']].iloc[i]
        act_exc = db_dict_code[removed_act_database, removed_act_code]
        if 'classifications' in act_exc:
            if 'CPC' in dict(act_exc['classifications']):
                if dict(act_exc['classifications'])['CPC'] in CPC_list:
                    # in carculator, the 'fuel' product may be diesel, NG or H2
                    if act_exc['reference product'] == 'fuel':
                        if 'cng' in act_exc['name']:
                            name_list.append('natural gas')
                        elif 'methane' in act_exc['name']:
                            name_list.append('natural gas')
                        elif 'diesel' in act_exc['name']:
                            name_list.append('diesel')
                        elif 'hydrogen' in act_exc['name']:
                            name_list.append('hydrogen')
                        else:
                            raise ValueError(f'Unknown fuel type in carculator for flow {row.Flow} in {row.Name}: '
                                             f'{act_exc["name"]}')
                    else:
                        name_list.append(act_exc['reference product'].split(',')[0])
                    unit_list.append(act_exc['unit'])

    if output_type == 'unit':
        if len(set(unit_list)) > 1:
            raise ValueError(f'Several units possible for flow {row.Flow} in {row.Name}: {set(unit_list)}')
        elif len(set(unit_list)) == 0:
            self.logger.warning(
                f'No flow found for type(s) {row.Flow} in {row.Name}. The efficiency of this technology cannot be '
                f'adjusted.'
            )
            return None
        else:
            return list(set(unit_list))[0]

    elif output_type == 'product':
        if len(set(name_list)) > 1:
            self.logger.warning(
                f'Several names possible for the same type of flow in {row.Name}: {set(name_list)}. '
                f'Kept the first one.')
            return list(set(name_list))[0]
        elif len(set(name_list)) == 0:
            self.logger.warning(
                f'No flow found for type(s) {row.Flow} in {row.Name}. The efficiency of this technology cannot be '
                f'adjusted.')
            return None
        else:
            return list(set(name_list))[0]

    else:
        raise ValueError(f'output_type must be either "unit" or "product"')

@staticmethod
def adapt_flows_to_efficiency_difference(
        act: dict,
        efficiency_ratio: float,
        techno_flows_to_correct: list[tuple[str, str]],
) -> dict:
    """
    Adapt the biosphere flows of an activity to correct the efficiency difference between ESM and LCA

    :param act: LCI dataset to adapt
    :param efficiency_ratio: ratio between the LCA and ESM efficiencies
    :param techno_flows_to_correct: list of (database, code) tuples to correct among the technosphere flows
    :return: the adapted LCI dataset
    """
    for exc in Dataset(act).get_biosphere_flows():
        if exc['unit'] in ['square meter-year', 'square meter']:
            # we exclude land occupation, land transformation elementary flows
            pass
        else:
            exc['amount'] *= efficiency_ratio
            exc['comment'] = (f'EF multiplied by {round(efficiency_ratio, 4)} (efficiency). '
                              + exc.get('comment', ''))

    for exc in Dataset(act).get_technosphere_flows():
        if (exc['database'], exc['code']) in techno_flows_to_correct:
            exc['amount'] *= efficiency_ratio
            exc['comment'] = (f'TF multiplied by {round(efficiency_ratio, 4)} (efficiency). '
                              + exc.get('comment', ''))

    act['comment'] = (f'Biosphere and fuel flows adjusted by a factor {round(efficiency_ratio, 4)} to correct '
                      f'efficiency difference between ESM and LCA. ' + act.get('comment', ''))

    return act


def get_lca_input_quantity(
        self,
        row: pd.Series,
        double_counting_removal_amount: pd.DataFrame
) -> float:
    """
    Retrieve the quantity of the flow removed from the double counting removal report

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param double_counting_removal_amount: dataframe containing the scaled amounts removed during double counting removal
    :return: quantity of the flow removed
    """
    amount = 0
    flows_list = row.Flow
    for flow in flows_list:
        try:
            amount += double_counting_removal_amount[(double_counting_removal_amount.Name == row.Name)
                                              & (double_counting_removal_amount.Flow == flow)].Amount.values[0]
        except IndexError:
            self.logger.warning(f'No flow of type {flow} has been removed in {row.Name}.')
            amount += 0
    else:
        return amount
