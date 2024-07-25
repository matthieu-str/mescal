from .utils import database_list_to_dict, get_biosphere_flows
import pandas as pd


def compute_efficiency_esm(row: pd.Series, model: pd.DataFrame) -> float:
    """
    Retrieve a technology efficiency from the ESM model file

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param model: input and output flows of the ESM model
    :return: efficiency of the process
    """
    try:
        input_amount = model[(model.Name == row.Name) & (model.Flow == row.Flow)].Amount.values[0]
    except IndexError:
        raise ValueError(f'The model has no technology {row.Name}, or the technology {row.Name} has no {row.Name} '
                         f'input flow.')
    else:
        return -1.0 / input_amount  # had negative sign as it is an input


def get_lca_input_quantity(row: pd.Series, double_counting_removal: pd.DataFrame) -> float:
    """
    Retrieve the quantity of the flow removed from the double counting removal report

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param double_counting_removal: dataframe containing the scaled amounts removed during double counting removal
    :return: quantity of the flow removed
    """
    try:
        amount = double_counting_removal[(double_counting_removal.Name == row.Name)
                                         & (double_counting_removal.Flow == row.Flow)].Amount.values[0]
    except IndexError:
        print(f'No flow of type {row.Flow} has been removed in {row.Name}. This technology will thus be removed.')
        return 0
    else:
        return amount


def get_lca_input_flow_unit_or_product(row: pd.Series, output_type: str, mapping_esm_flows_to_CPC: pd.DataFrame,
                                       db_dict_name: dict, removed_flows: pd.DataFrame) -> str | None:
    """
    Retrieve the unit or product of the flow removed from the double counting removal report

    :param row: series containing the name of the ESM technology and the name of the ESM flow to check for efficiency
        difference
    :param output_type: can be either 'unit' or 'product'
    :param mapping_esm_flows_to_CPC: dataframe containing the mapping between ESM flows and CPC categories
    :param db_dict_name: LCI database as a dictionary with (name, product, location, database) as keys
    :param removed_flows: dataframe containing the name and amount of removed flows during double counting removal
    :return: the unit or product of the flow removed for the ESM (technology, flow) couple
    """
    unit_list = []
    name_list = []
    CPC_list = mapping_esm_flows_to_CPC[mapping_esm_flows_to_CPC['Flow'] == row.Flow]['CPC'].values[0]
    df_removed_flows = removed_flows[removed_flows.Name == row.Name]

    for i in range(len(df_removed_flows)):
        removed_act_name, removed_act_prod, removed_act_loc, removed_act_database = df_removed_flows[
            ['Removed flow activity', 'Removed flow product', 'Removed flow location', 'Removed flow database']].iloc[i]
        act_exc = db_dict_name[removed_act_name, removed_act_prod, removed_act_loc, removed_act_database]
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
            print(f'No flow found for type {row.Flow} in {row.Name}. This technology will thus be removed.')
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
            print(f'No flow found for type {row.Flow} in {row.Name}. This technology will thus be removed.')
            return None
        else:
            return list(set(name_list))[0]

    else:
        raise ValueError(f'output_type must be either "unit" or "product"')


def adapt_biosphere_flows_to_efficiency_difference(act: dict, efficiency_ratio: float) -> dict:
    """
    Adapt the biosphere flows of an activity to correct the efficiency difference between ESM and LCA

    :param act: LCI dataset to adapt
    :param efficiency_ratio: ratio between the LCA and ESM efficiencies
    :return: the adapted LCI dataset
    """
    for exc in get_biosphere_flows(act):
        if exc['unit'] in ['square meter-year', 'square meter', 'megajoule']:
            # we exclude land occupation, land transformation and energy elementary flows
            pass
        else:
            exc['amount'] *= efficiency_ratio
            exc['comment'] = (f'EF multiplied by {efficiency_ratio} (efficiency).' + act.get('comment', ''))
    return act


def correct_esm_and_lca_efficiency_differences(db: list[dict], model: pd.DataFrame, efficiency: pd.DataFrame,
                                               mapping_esm_flows_to_CPC: pd.DataFrame, removed_flows: pd.DataFrame,
                                               unit_conversion: pd.DataFrame, double_counting_removal: pd.DataFrame,
                                               output_type: str = 'database') \
        -> list[dict] | pd.DataFrame | tuple[list[dict], pd.DataFrame]:
    """
    Correct the efficiency differences between ESM and LCA for the technologies in the database

    :param db: LCI database as a list of dictionaries
    :param model: model file containing the input and output flows of the ESM model
    :param efficiency: file containing the ESM (Name, Flow) couples to correct
    :param mapping_esm_flows_to_CPC: mapping between ESM flows and CPC categories
    :param removed_flows: dataframe containing the name and amount of removed flows during double counting removal
    :param unit_conversion: dataframe containing the conversion factors between different units
    :param double_counting_removal: dataframe containing the scaled amounts removed during double counting removal
    :param output_type: can be either 'database', 'dataframe', or 'both'. If 'database', the corrected LCI database is
        returned. If 'dataframe', the efficiency dataframe is returned. If 'both', both are returned
        (database, dataframe).
    :return: the corrected LCI database
    """
    db_dict_name = database_list_to_dict(db, 'name')

    efficiency['ESM efficiency'] = efficiency.apply(compute_efficiency_esm, axis=1, model=model)
    efficiency['LCA input unit'] = efficiency.apply(get_lca_input_flow_unit_or_product, axis=1, output_type='unit',
                                                    mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC,
                                                    db_dict_name=db_dict_name, removed_flows=removed_flows)
    efficiency.drop(efficiency[efficiency['LCA input unit'].isnull()].index, inplace=True)
    efficiency['LCA input product'] = efficiency.apply(get_lca_input_flow_unit_or_product, axis=1,
                                                       output_type='product',
                                                       mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC,
                                                       db_dict_name=db_dict_name, removed_flows=removed_flows)
    efficiency.drop(efficiency[efficiency['LCA input product'].isnull()].index, inplace=True)
    efficiency['LCA input quantity'] = efficiency.apply(get_lca_input_quantity, axis=1,
                                                        double_counting_removal=double_counting_removal)
    efficiency.drop(efficiency[efficiency['LCA input quantity'] == 0].index, inplace=True)
    efficiency = efficiency.merge(unit_conversion[unit_conversion.Type == 'Operation'][['Name', 'Value', 'LCA', 'ESM']],
                                  how='left', on='Name')
    efficiency.rename(columns={'Value': 'Output conversion factor', 'LCA': 'LCA output unit', 'ESM': 'ESM output unit'},
                      inplace=True)
    efficiency = efficiency.merge(
        unit_conversion[(unit_conversion.Type == 'Other') & (unit_conversion.ESM == 'kilowatt hour')][
            ['Name', 'Value', 'LCA']], how='left', left_on=['LCA input product', 'LCA input unit'],
        right_on=['Name', 'LCA'], suffixes=('', '_to_remove')
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
        tech, flow = efficiency[['Name', 'Flow']].iloc[i]
        CPC_list = mapping_esm_flows_to_CPC[mapping_esm_flows_to_CPC['Flow'] == flow]['CPC'].values[0]
        df_removed_flows = removed_flows[removed_flows.Name == tech]
        for j in range(len(df_removed_flows)):
            (main_act_name, main_act_prod, main_act_loc, main_act_database, removed_act_name, removed_act_prod,
             removed_act_loc, removed_act_database) = df_removed_flows[
                ['Activity', 'Product', 'Location', 'Database', 'Removed flow activity', 'Removed flow product',
                 'Removed flow location', 'Removed flow database']].iloc[j]
            act_exc = db_dict_name[removed_act_name, removed_act_prod, removed_act_loc, removed_act_database]
            if 'classifications' in act_exc:
                if 'CPC' in dict(act_exc['classifications']):
                    if dict(act_exc['classifications'])['CPC'] in CPC_list:
                        act_to_adapt = db_dict_name[main_act_name, main_act_prod, main_act_loc, main_act_database]
                        act_to_adapt_list.append(act_to_adapt)
        for act in act_to_adapt_list:
            efficiency_ratio = efficiency['LCA efficiency'].iloc[i] / efficiency['ESM efficiency'].iloc[i]
            act = adapt_biosphere_flows_to_efficiency_difference(act, efficiency_ratio)

    if output_type == 'dataframe':
        return efficiency
    elif output_type == 'database':
        return db
    elif output_type == 'both':
        return db, efficiency
