import pandas as pd
import ast
from .database import Dataset


def correct_esm_and_lca_efficiency_differences(
        self,
        removed_flows: pd.DataFrame,
        double_counting_removal: pd.DataFrame,
) -> None:
    """
    Correct the efficiency differences between ESM and LCA for the technologies in the database

    :param removed_flows: dataframe containing the name, code and amount of removed flows during double counting
        removal
    :param double_counting_removal: dataframe containing the scaled amounts removed during double counting removal
    :return: None
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    db_dict_code = self.main_database.db_as_dict_code
    efficiency = self.efficiency
    unit_conversion = self.unit_conversion
    mapping_esm_flows_to_CPC_cat = self.mapping_esm_flows_to_CPC_cat

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
        double_counting_removal=double_counting_removal
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

    for i in range(len(efficiency)):

        act_to_adapt_list = []
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
                        act_to_adapt = db_dict_code[main_act_database, main_act_code]
                        act_to_adapt_list.append(act_to_adapt)

        if len(act_to_adapt_list) == 0:
            print(f'No flow of type(s) {flows_list} found for {tech}. The efficiency of this technology cannot be '
                  f'adjusted.')

        for act in act_to_adapt_list:
            efficiency_ratio = efficiency['LCA efficiency'].iloc[i] / efficiency['ESM efficiency'].iloc[i]
            act = self.adapt_biosphere_flows_to_efficiency_difference(act, efficiency_ratio, tech)


def compute_efficiency_esm(
        self,
        row: pd.Series,
) -> float:
    """
    Retrieve a technology efficiency from the ESM model file

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :return: efficiency of the process
    """
    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    model = self.model

    flows_list = row.Flow
    input_amount = 0

    for flow in flows_list:
        try:
            input_amount += model[(model.Name == row.Name) & (model.Flow == flow)].Amount.values[0]
        except IndexError:
            raise ValueError(f'The model has no technology {row.Name}, or the technology {row.Name} has no {flow} '
                             f'input flow.')
    return -1.0 / input_amount  # had negative sign as it is an input


def get_lca_input_flow_unit_or_product(
        self,
        row: pd.Series,
        output_type: str,
        removed_flows: pd.DataFrame
) -> str | None:
    """
    Retrieve the unit or product of the flow removed from the double counting removal report

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
                        elif 'diesel' in act_exc['name']:
                            name_list.append('diesel')
                        elif 'hydrogen' in act_exc['name']:
                            name_list.append('hydrogen')
                        else:
                            raise ValueError(f'Unknown fuel type for flow {row.Flow} in {row.Name}')
                    else:
                        name_list.append(act_exc['reference product'].split(',')[0])
                    unit_list.append(act_exc['unit'])

    if output_type == 'unit':
        if len(set(unit_list)) > 1:
            raise ValueError(f'Several units possible for flow {row.Flow} in {row.Name}: {set(unit_list)}')
        elif len(set(unit_list)) == 0:
            print(
                f'No flow found for type(s) {row.Flow} in {row.Name}. The efficiency of this technology cannot be '
                f'adjusted.')
            return None
        else:
            return list(set(unit_list))[0]

    elif output_type == 'product':
        if len(set(name_list)) > 1:
            print(
                f'Several names possible for the same type of flow in {row.Name}: {set(name_list)}. '
                f'Kept the first one.')
            return list(set(name_list))[0]
        elif len(set(name_list)) == 0:
            print(
                f'No flow found for type(s) {row.Flow} in {row.Name}. The efficiency of this technology cannot be '
                f'adjusted.')
            return None
        else:
            return list(set(name_list))[0]

    else:
        raise ValueError(f'output_type must be either "unit" or "product"')


def adapt_biosphere_flows_to_efficiency_difference(
        act: dict,
        efficiency_ratio: float,
        tech: str
) -> dict:
    """
    Adapt the biosphere flows of an activity to correct the efficiency difference between ESM and LCA

    :param act: LCI dataset to adapt
    :param efficiency_ratio: ratio between the LCA and ESM efficiencies
    :param tech: name of the technology
    :return: the adapted LCI dataset
    """
    for exc in Dataset(act).get_biosphere_flows():
        if exc['unit'] in ['square meter-year', 'square meter', 'megajoule']:
            # we exclude land occupation, land transformation and energy elementary flows
            pass
        else:
            exc['amount'] *= efficiency_ratio
            exc['comment'] = (f'{tech}: EF multiplied by {round(efficiency_ratio, 4)} (efficiency). '
                              + exc.get('comment', ''))

    act['comment'] = (f'Biosphere flows adjusted by a factor {round(efficiency_ratio, 4)} to correct efficiency '
                      f'difference between ESM and LCA. ' + act.get('comment', ''))
    return act


def get_lca_input_quantity(
        row: pd.Series,
        double_counting_removal: pd.DataFrame
) -> float:
    """
    Retrieve the quantity of the flow removed from the double counting removal report

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param double_counting_removal: dataframe containing the scaled amounts removed during double counting removal
    :return: quantity of the flow removed
    """
    amount = 0
    flows_list = row.Flow
    for flow in flows_list:
        try:
            amount += double_counting_removal[(double_counting_removal.Name == row.Name)
                                              & (double_counting_removal.Flow == flow)].Amount.values[0]
        except IndexError:
            print(f'No flow of type {flow} has been removed in {row.Name}.')
            amount += 0
    else:
        return amount
