import pandas as pd

from .regionalization import *


def create_or_modify_activity_from_esm_results(db: list[dict], original_activity_prod: str, original_activity_name: str,
                                               original_activity_database: str, flows: pd.DataFrame,
                                               model: pd.DataFrame, esm_location: str, esm_results: pd.DataFrame,
                                               mapping: pd.DataFrame, accepted_locations: list[str],
                                               locations_ranking: list[str],
                                               unit_conversion: pd.DataFrame) -> list[dict]:

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

    unit_conversion['From'] = unit_conversion['From'].apply(ecoinvent_unit_convention)
    unit_conversion['To'] = unit_conversion['To'].apply(ecoinvent_unit_convention)

    model['Flow'] = model.apply(replace_mobility_end_use_type, axis=1)
    model = pd.merge(model, model[model.Amount == 1.0].drop(columns=['Amount']).rename(columns={'Flow': 'Output'}),
                     how='left', on='Name')

    act_to_flows_dict = {(flows['Product'].iloc[i], flows['Activity'].iloc[i]): list(
        flows[(flows['Product'] == flows['Product'].iloc[i])
              & (flows['Activity'] == flows['Activity'].iloc[i])]['Name']
    ) for i in range(len(flows))}

    flows_list = act_to_flows_dict[(original_activity_prod, original_activity_name)]
    end_use_tech_list = list(model[model.Output.isin(flows_list)].Name.unique())

    if set(flows_list) == {'ELECTRICITY_EHV', 'ELECTRICITY_HV'}:
        # the high and extra high voltage electricity are merged in the LCI database,
        # thus we remove transformations between these two levels of voltage
        end_use_tech_list.remove('TRAFO_HE')
        end_use_tech_list.remove('TRAFO_EH')
    elif flows_list == ['ELECTRICITY_LV']:
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_ELEC')
    elif set(flows_list) == {'NG_HP', 'NG_EHP'}:
        # the high and extra high pressure natural gas are merged in the LCI database,
        # thus we remove transformations between these two levels of pressure
        end_use_tech_list.remove('NG_EXP_EH')
        end_use_tech_list.remove('NG_EXP_EH_COGEN')
        end_use_tech_list.remove('NG_COMP_HE')
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_NG')
    elif set(flows_list) == {'H2_LP', 'H2_MP', 'H2_HP', 'H2_EHP'}:
        # all pressure levels for hydrogen are merged in the LCI database,
        # thus we remove transformations between these two levels of pressure
        end_use_tech_list.remove('H2_COMP_HE')
        end_use_tech_list.remove('H2_COMP_MH')
        end_use_tech_list.remove('H2_COMP_LM')
        end_use_tech_list.remove('H2_EXP_EH')
        end_use_tech_list.remove('H2_EXP_HM')
        end_use_tech_list.remove('H2_EXP_ML')
        end_use_tech_list.remove('H2_EXP_EH_COGEN')
        end_use_tech_list.remove('H2_EXP_HM_COGEN')
        end_use_tech_list.remove('H2_EXP_ML_COGEN')
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_H2')
    elif flows_list == ['GASOLINE']:
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_GASO')
    elif flows_list == ['DIESEL']:
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_DIE')
    elif flows_list == ['CO2_C']:
        # the storage technologies should be removed (production only)
        end_use_tech_list.remove('STO_CO2')
    else:
        pass

    total_amount = 0  # initialize the total amount of production

    for tech in end_use_tech_list:
        if tech in list(esm_results.Name.unique()):
            amount = esm_results[esm_results.Name == tech].Production.iloc[0]
        else:  # if the technology is not in the ESM results, we assume that its production is null
            amount = 0
        total_amount += amount

    exchanges = []

    for tech in end_use_tech_list:
        if tech in list(esm_results.Name.unique()):
            amount = esm_results[esm_results.Name == tech].Production.iloc[0]
        else:
            amount = 0
        if amount == 0:
            pass
        else:
            if tech in list(mapping[(mapping.Type == 'Operation') | (mapping.Type == 'Resource')].Name.unique()):
                activity_name = mapping[
                    (mapping.Name == tech) & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                                        ].Activity.iloc[0]
                activity_prod = mapping[
                    (mapping.Name == tech) & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                                        ].Product.iloc[0]
                activity_database = mapping[
                    (mapping.Name == tech) & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                                            ].Database.iloc[0]
                activity_location = mapping[
                    (mapping.Name == tech) & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                                            ].Location.iloc[0]
                activity_unit = db_dict_name[activity_name, activity_prod, activity_location, activity_database]['unit']

                if activity_unit != original_activity_unit:
                    conversion_factor = unit_conversion[
                        (unit_conversion.Name.apply(lambda x: x in original_activity_prod))
                        & (unit_conversion.From == activity_unit)
                        & (unit_conversion.To == original_activity_unit)
                        ].Value.iloc[0]
                    amount *= conversion_factor

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
                    'comment': tech,
                }
                exchanges.append(new_exc)
            else:
                print(f'The technology {tech} is not in the mapping file. '
                      f'It cannot be considered in the result LCI dataset.')

    exchanges.append(
        {
            'amount': 1.0,
            'code': new_code,
            'type': 'production',
            'name': original_activity_name,
            'product': original_activity_prod,
            'unit': original_activity_unit,
            'location': esm_location,
            'database': original_activity_database,
        }
    )

    # Add non-production flows to the new activity
    for exc in activity['exchanges']:
        if exc['unit'] != original_activity_unit:
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

    return db


def replace_mobility_end_use_type(row):
    if ('BUS' in row['Name']) & ('MOB_PUBLIC' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_PUBLIC_BUS'
    elif ('COACH' in row['Name']) & ('MOB_PUBLIC' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_PUBLIC_COACH'
    elif ('TRAIN' in row['Name']) & ('MOB_PUBLIC' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_PUBLIC_TRAIN'
    elif ('CAR' in row['Name']) & ('MOB_PRIVATE' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_PRIVATE_CAR'
    elif ('SUV' in row['Name']) & ('MOB_PRIVATE' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_PRIVATE_SUV'
    elif ('LCV' in row['Name']) & ('MOB_FREIGHT' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_FREIGHT_LCV'
    elif ('SEMI' in row['Name']) & ('MOB_FREIGHT' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_FREIGHT_SEMI'
    elif ('TRUCK' in row['Name']) & ('MOB_FREIGHT' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_FREIGHT_TRUCK'
    elif ('TRAIN' in row['Name']) & ('MOB_FREIGHT' in row['Flow']) & (row['Amount'] == 1.0):
        return 'MOB_FREIGHT_TRAIN'
    else:
        return row['Flow']


def ecoinvent_unit_convention(unit):
    unit_dict = {
        'kg': 'kilogram',
        'm2': 'square meter',
        'm3': 'cubic meter',
        'MJ': 'megajoule',
        'kWh': 'kilowatt hour',
        'h': 'hour',
        'km': 'kilometer',
        'pkm': 'person kilometer',
        'tkm': 'ton kilometer',
    }
    if unit in unit_dict:
        return unit_dict[unit]
    else:
        return unit


def create_new_database_with_esm_results(mapping, model, esm_location, esm_results, locations_ranking,
                                         accepted_locations, unit_conversion, db, new_db_name):

    flows = mapping[mapping.Type == 'Flow']

    already_done = []

    # Create the new LCI datasets from the ESM results
    for i in range(len(flows)):

        original_activity_prod = flows.Product.iloc[i]
        original_activity_name = flows.Activity.iloc[i]
        original_activity_database = flows.Database.iloc[i]

        if (original_activity_name, original_activity_prod, esm_location, original_activity_database) in already_done:
            pass

        else:
            db = create_or_modify_activity_from_esm_results(
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
            )

            already_done.append((original_activity_name,
                                 original_activity_prod,
                                 esm_location,
                                 original_activity_database))

    db_dict_name = database_list_to_dict(db, 'name')

    # Plugging the new activity in the database
    for i in range(len(flows)):

        original_activity_name = flows.Activity.iloc[i]
        activity_prod = flows.Product.iloc[i]
        activity_database = flows.Database.iloc[i]

        new_activity = db_dict_name[
            original_activity_name + ', from ESM results',
            activity_prod,
            esm_location,
            activity_database
        ]

        # Activities of the ESM region
        activities_of_esm_region = [a for a in wurst.get_many(db, *[wurst.equals('location', esm_location)])]
        for act in activities_of_esm_region:
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

    write_wurst_database_to_brightway(db, new_db_name)
