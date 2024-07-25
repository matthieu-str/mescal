from .double_counting import *


def create_or_modify_activity_from_esm_results(db: list[dict], original_activity_prod: str, original_activity_name: str,
                                               original_activity_database: str, flows: pd.DataFrame,
                                               model: pd.DataFrame, esm_location: str, esm_results: pd.DataFrame,
                                               mapping: pd.DataFrame, accepted_locations: list[str],
                                               locations_ranking: list[str], tech_to_remove_layers: pd.DataFrame,
                                               unit_conversion: pd.DataFrame, new_end_use_types: pd.DataFrame) \
        -> tuple[list[dict], list[list[str]]]:
    """
    Create or modify an activity in the LCI database based on the ESM results

    :param db: list of activities in the LCI database
    :param original_activity_prod: reference product of the original activity
    :param original_activity_name: name of the original activity
    :param original_activity_database: database of the original activity
    :param flows: mapping file between energy flows and LCI datasets
    :param model: inputs and outputs of the ESM technologies
    :param esm_location: ecoinvent location under study in the ESM
    :param esm_results: file of the ESM results in terms of annual production
    :param mapping: mapping file between ESM technologies and LCI datasets
    :param accepted_locations: list of accepted locations for the regionalization
    :param locations_ranking: ranking of preferred locations for the regionalization
    :param tech_to_remove_layers: technologies to remove from the result LCI datasets
    :param unit_conversion: unit conversion factors
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :return: the updated LCI database, list of activities to perform double counting removal
    """
    db_dict_name = database_list_to_dict(db, 'name')

    # Check if the original activity is in the database for the location under study
    if (original_activity_name, original_activity_prod, esm_location, original_activity_database) in db_dict_name:
        activity = db_dict_name[
            original_activity_name, original_activity_prod, esm_location, original_activity_database
        ]

    # If not, we take a similar activity with another location and regionalize its foreground
    else:
        activity = [a for a in wurst.get_many(db, *[
            wurst.equals('name', original_activity_name),
            wurst.equals('reference product', original_activity_prod),
            wurst.equals('database', original_activity_database)
        ])][0]

        activity = regionalize_activity_foreground(
            act=activity,
            accepted_locations=accepted_locations,
            target_region=esm_location,
            locations_ranking=locations_ranking,
            db=db,
            db_dict_code=database_list_to_dict(db, 'code'),
            db_dict_name=database_list_to_dict(db, 'name')
        )

    new_code = random_code()
    original_activity_unit = activity['unit']
    prod_flow = get_production_flow(activity)
    prod_flow_amount = prod_flow['amount']  # can be -1 if it is a waste activity

    unit_conversion['LCA'] = unit_conversion['LCA'].apply(ecoinvent_unit_convention)
    unit_conversion['ESM'] = unit_conversion['ESM'].apply(ecoinvent_unit_convention)

    model['Flow'] = model.apply(lambda x: replace_mobility_end_use_type(row=x,
                                                                        new_end_use_types=new_end_use_types), axis=1)
    model = pd.merge(model, model[model.Amount == 1.0].drop(columns=['Amount']).rename(columns={'Flow': 'Output'}),
                     how='left', on='Name')

    act_to_flows_dict = {(flows['Product'].iloc[i], flows['Activity'].iloc[i]): list(
        flows[(flows['Product'] == flows['Product'].iloc[i])
              & (flows['Activity'] == flows['Activity'].iloc[i])]['Name']
    ) for i in range(len(flows))}

    flows_list = act_to_flows_dict[(original_activity_prod, original_activity_name)]
    end_use_tech_list = list(model[model.Output.isin(flows_list)].Name.unique())

    if len(end_use_tech_list) == 0:
        # Case where the layer has no production
        return db, []

    try:
        tech_to_remove_layers['Layers'] = tech_to_remove_layers['Layers'].apply(ast.literal_eval)
        tech_to_remove_layers['Technologies'] = tech_to_remove_layers['Technologies'].apply(ast.literal_eval)
    except ValueError:
        pass

    for i in range(len(tech_to_remove_layers)):
        if set(flows_list) == set(tech_to_remove_layers.Layers.iloc[i]):
            for tech in tech_to_remove_layers.Technologies.iloc[i]:
                end_use_tech_list.remove(tech)
        else:
            pass

    total_amount = 0  # initialize the total amount of production
    check_layers_mapping = False

    for tech in end_use_tech_list:

        if tech in list(esm_results.Name.unique()):
            amount = esm_results[esm_results.Name == tech].Production.iloc[0]

        else:  # if the technology is not in the ESM results, we assume that its production is null
            amount = 0

        if tech in list(mapping[(mapping.Type == 'Operation') | (mapping.Type == 'Resource')].Name.unique()):
            total_amount += amount
            check_layers_mapping = True
        else:
            pass  # if the technology is not in the mapping file, we do not consider it in the result LCI dataset

    if check_layers_mapping is False:
        raise ValueError(f'The layer {flows_list} does not have any technology in the mapping file.')

    if total_amount == 0:  # no production in the layer
        return db, []

    exchanges = []
    perform_d_c = []

    for tech in end_use_tech_list:
        if tech in list(esm_results.Name.unique()):
            amount = esm_results[esm_results.Name == tech].Production.iloc[0]
        else:
            amount = 0
        if amount == 0:
            pass
        else:
            if tech in list(mapping[(mapping.Type == 'Operation') | (mapping.Type == 'Resource')].Name.unique()):
                activity_name, activity_prod, activity_database, activity_location = mapping[
                    (mapping.Name == tech) & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                ][['Activity', 'Product', 'Database', 'Location']].values[0]

                activity_unit = db_dict_name[activity_name, activity_prod, activity_location, activity_database]['unit']

                if activity_unit != original_activity_unit:
                    if original_activity_prod.split(',')[0] == 'transport':
                        conversion_factor = unit_conversion[
                            (unit_conversion.Name == tech)
                            & (unit_conversion.ESM == original_activity_unit)
                            & (unit_conversion.LCA == activity_unit)
                            ].Value.values
                    else:
                        conversion_factor = unit_conversion[
                            (unit_conversion.Name == original_activity_prod.split(',')[0])
                            & (unit_conversion.ESM == original_activity_unit)
                            & (unit_conversion.LCA == activity_unit)
                            ].Value.values
                    if len(list(set(conversion_factor))) == 0:
                        raise ValueError(f'The unit conversion factor between {activity_unit} and '
                                         f'{original_activity_unit} for {original_activity_prod.split(",")[0]} '
                                         f'is not in the unit conversion file.')
                    elif len(list(set(conversion_factor))) > 1:
                        raise ValueError(f'Multiple possible conversion factors between {activity_unit} and '
                                         f'{original_activity_unit} for {original_activity_prod.split(",")[0]}')
                    else:
                        amount *= conversion_factor[0]
                else:
                    conversion_factor = 1.0

                if prod_flow_amount == -1.0:  # for waste activities
                    amount *= -1.0

                code = db_dict_name[activity_name, activity_prod, activity_location, activity_database]['code']
                new_exc = {
                    'amount': amount / total_amount,
                    'code': code,
                    'type': 'technosphere',
                    'name': activity_name,
                    'product': activity_prod,
                    'unit': activity_unit,
                    'location': activity_location,
                    'database': activity_database,
                    'comment': f'{tech}, {conversion_factor}',
                }
                exchanges.append(new_exc)
                if tech in list(mapping[mapping.Type == 'Operation'].Name.unique()):
                    # we only perform double counting removal for the operation activities
                    perform_d_c.append([tech, activity_prod, activity_name, activity_location, activity_database, code])
            else:
                print(f'The technology {tech} is not in the mapping file. '
                      f'It cannot be considered in the result LCI dataset.')

    exchanges.append(
        {
            'amount': prod_flow_amount,
            'code': new_code,
            'type': 'production',
            'name': original_activity_name,
            'product': original_activity_prod,
            'unit': original_activity_unit,
            'location': esm_location,
            'database': original_activity_database,
        }
    )

    for exc in activity['exchanges']:
        if exc['unit'] != original_activity_unit:  # Add non-production flows to the new activity
            exchanges.append(exc)
        else:
            pass

    new_activity = {
        'database': original_activity_database,
        'name': original_activity_name + ', from ESM results',
        'location': esm_location,
        'unit': original_activity_unit,
        'reference product': original_activity_prod,
        'code': new_code,
        'classifications': activity['classifications'],
        'comment': f'Activity derived from the ESM results in the layers {flows_list} for {esm_location}. '
                   + activity['comment'],
        'parameters': activity.get('parameters', {}),
        'categories': activity.get('categories', None),
        'exchanges': exchanges,
    }

    db.append(new_activity)

    return db, perform_d_c


def replace_mobility_end_use_type(row: pd.Series, new_end_use_types: pd.DataFrame) -> str:
    """
    Reformat the end use type of the mobility technologies

    :param row: row of the model dataframe
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :return: updated end use type
    """

    for i in range(len(new_end_use_types)):
        name = new_end_use_types.Name.iloc[i]
        old_eut = new_end_use_types.Old.iloc[i]
        new_eut = new_end_use_types.New.iloc[i]
        search_type = new_end_use_types['Search type'].iloc[i]
        if search_type == 'startswith':
            if (row['Name'].startswith(name)) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        elif search_type == 'contains':
            if (name in row['Name']) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        elif search_type == 'equals':
            if (name == row['Name']) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        else:
            raise ValueError('The search type should be either "startswith", "contains" or "equals".')

    return row['Flow']


def create_new_database_with_esm_results(mapping: pd.DataFrame, model: pd.DataFrame, esm_location: str,
                                         esm_results: pd.DataFrame, locations_ranking: list[str],
                                         accepted_locations: list[str], unit_conversion: pd.DataFrame,
                                         db: list[dict], mapping_esm_flows_to_CPC_cat: pd.DataFrame,
                                         new_end_use_types: pd.DataFrame = None, new_db_name: str = 'default',
                                         tech_specifics: pd.DataFrame = None,
                                         technology_compositions: pd.DataFrame = None,
                                         tech_to_remove_layers: pd.DataFrame = None, write_database: bool = True,
                                         return_obj: str = None) -> None | list[dict]:
    """
    Create a new database with the ESM results

    :param mapping: mapping file between ESM technologies and LCI datasets
    :param model: inputs and outputs of the ESM technologies
    :param esm_location: location under study in the ESM
    :param esm_results: results of the ESM in terms of annual production
    :param locations_ranking: ranking of preferred locations for the regionalization
    :param accepted_locations: list of accepted locations for the regionalization
    :param unit_conversion: unit conversion factors
    :param db: list of activities in the LCI database
    :param new_db_name: name of the new database. By default, it is the name of the original database with
        '_with_ESM_results_for_{esm_location}'
    :param tech_specifics: technology-specific information
    :param technology_compositions: technology compositions
    :param tech_to_remove_layers: technologies to remove from the result LCI datasets
    :param mapping_esm_flows_to_CPC_cat: mapping file between ESM flows and CPC categories
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :param write_database: if True, write the new database in the brightway2 project
    :param return_obj: if 'database', return the new database, else does not return anything
    :return: None or the new database as a list of dictionaries if 'return_obj' is 'database
    """

    if tech_specifics is None:
        tech_specifics = pd.DataFrame(columns=['Name', 'Specifics', 'Amount'])
    if technology_compositions is None:
        technology_compositions = pd.DataFrame(columns=['Name', 'Components'])
    if tech_to_remove_layers is None:
        tech_to_remove_layers = pd.DataFrame(columns=['Layers', 'Technologies'])
    if new_end_use_types is None:
        new_end_use_types = pd.DataFrame(columns=['Name', 'Search type', 'Old', 'New'])
    if new_db_name == 'default':
        new_db_name = f"{db[0]['database']}_with_ESM_results_for_{esm_location}"

    flows = mapping[mapping.Type == 'Flow']

    already_done = []
    perform_d_c = []

    # Create the new LCI datasets from the ESM results
    for i in range(len(flows)):

        original_activity_prod = flows.Product.iloc[i]
        original_activity_name = flows.Activity.iloc[i]
        original_activity_database = flows.Database.iloc[i]

        if (original_activity_name, original_activity_prod, esm_location, original_activity_database) in already_done:
            pass

        else:
            db, new_perform_d_c = create_or_modify_activity_from_esm_results(
                db=db,
                original_activity_prod=original_activity_prod,
                original_activity_name=original_activity_name,
                original_activity_database=original_activity_database,
                model=model,
                flows=flows,
                esm_location=esm_location,
                esm_results=esm_results,
                mapping=mapping,
                accepted_locations=accepted_locations,
                locations_ranking=locations_ranking,
                unit_conversion=unit_conversion,
                tech_to_remove_layers=tech_to_remove_layers,
                new_end_use_types=new_end_use_types,
            )

            already_done.append((original_activity_name,
                                 original_activity_prod,
                                 esm_location,
                                 original_activity_database))

            perform_d_c += new_perform_d_c

    db_dict_name = database_list_to_dict(db, 'name')
    db_dict_code = database_list_to_dict(db, 'code')

    # Plugging the new activity in the database
    for i in range(len(flows)):

        original_activity_name = flows.Activity.iloc[i]
        activity_prod = flows.Product.iloc[i]
        activity_database = flows.Database.iloc[i]

        if (
                original_activity_name + ', from ESM results',
                activity_prod,
                esm_location,
                activity_database
        ) in db_dict_name:
            # if not, it means that the activity has not been created in the previous step
            # (e.g., no production, trivial results)

            new_activity = db_dict_name[
                original_activity_name + ', from ESM results',
                activity_prod,
                esm_location,
                activity_database
            ]

            # Activities of the ESM region
            activities_of_esm_region = [a for a in wurst.get_many(db, *[wurst.equals('location', esm_location)])]
            for act in activities_of_esm_region:
                if act['name'] == original_activity_name + ', from ESM results':
                    pass  # we do not want the new activity to be an input of itself
                else:
                    for exc in get_technosphere_flows(act):
                        if (
                                (exc['name'] == original_activity_name)
                                & (exc['product'] == activity_prod)
                        ):
                            exc['code'] = new_activity['code']
                            exc['location'] = esm_location
                        else:
                            pass

            # Downstream activities of the original activity, if it exists for the ESM location
            if (original_activity_name, activity_prod, esm_location, activity_database) in db_dict_name:
                original_activity = db_dict_name[
                    original_activity_name, activity_prod, esm_location, activity_database
                ]
                downstream_consumers = get_downstream_consumers(original_activity, db)
                for act in downstream_consumers:
                    if act['name'] == original_activity_name + ', from ESM results':
                        pass  # we do not want the new activity to be an input of itself
                    else:
                        for exc in get_technosphere_flows(act):
                            if (
                                    (exc['name'] == original_activity_name)
                                    & (exc['product'] == activity_prod)
                                    & (exc['location'] == esm_location)
                            ):
                                exc['code'] = new_activity['code']
                                exc['location'] = esm_location
                            else:
                                pass

    # Double counting removal of the construction activities
    double_counting_act = pd.DataFrame(data=perform_d_c,
                                       columns=['Name', 'Product', 'Activity', 'Location', 'Database', 'Current_code'])

    # Adding current code to the mapping file
    mapping['Current_code'] = mapping.apply(lambda row: get_code(
        db_dict_name=db_dict_name,
        product=row['Product'],
        activity=row['Activity'],
        region=row['Location'],
        database=row['Database']
    ), axis=1)

    # mapping['New_code'] = mapping.apply(lambda row: random_code(), axis=1)
    # double_counting_act['New_code'] = double_counting_act.apply(lambda row: random_code(), axis=1)
    mapping_constr = mapping[mapping.Type == 'Construction']

    model = model.pivot(index='Name', columns='Flow', values='Amount').reset_index()
    model.fillna(0, inplace=True)

    N = double_counting_act.shape[1]

    try:
        technology_compositions_dict = {key: ast.literal_eval(value) for key, value in dict(zip(
            technology_compositions.Name, technology_compositions.Components
        )).items()}
    except ValueError:
        technology_compositions_dict = dict(zip(technology_compositions.Name, technology_compositions.Components))

    double_counting_act = pd.merge(double_counting_act, model, on='Name', how='left')
    double_counting_act['CONSTRUCTION'] = double_counting_act.shape[0] * [0]
    (double_counting_act, background_search_act, no_construction_list,
     no_background_search_list) = add_technology_specifics(double_counting_act, tech_specifics)

    db, db_dict_code, db_dict_name, flows_set_to_zero, ei_removal = double_counting_removal(
        df_op=double_counting_act,
        df_constr=mapping_constr,
        esm_db_name=db[0]['database'],
        mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC_cat,
        technology_compositions_dict=technology_compositions_dict,
        db=db,
        db_dict_code=db_dict_code,
        db_dict_name=db_dict_name,
        N=N,
        background_search_act=background_search_act,
        no_construction_list=no_construction_list,
        no_background_search_list=no_background_search_list,
        ESM_inputs=['OWN_CONSTRUCTION', 'CONSTRUCTION'],
        create_new_db=False,
    )

    if write_database:
        write_wurst_database_to_brightway(db, new_db_name)
    if return_obj == 'database':
        db = change_database_name(db, new_db_name)
        return db
