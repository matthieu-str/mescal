import pandas as pd
from .utils import *


def create_or_modify_activity_from_esm_results(db: list[dict], original_activity_prod: str, original_activity_name: str,
                                               original_activity_database: str, flows: pd.DataFrame,
                                               model: pd.DataFrame, esm_results: pd.DataFrame,
                                               mapping: pd.DataFrame) -> list[dict]:

    db_dict_name = database_list_to_dict(db, 'name')

    model = pd.merge(model, model[model.Amount == 1.0].drop(columns=['Amount']).rename(columns={'Flow': 'Output'}),
                     how='left', on='Name')

    act_to_flows_dict = {(flows['Product'].iloc[i], flows['Activity'].iloc[i]): list(
        flows[(flows['Product'] == flows['Product'].iloc[i])
              & (flows['Activity'] == flows['Activity'].iloc[i])]['Name']
    ) for i in range(len(flows))}

    flows_list = act_to_flows_dict[(original_activity_prod, original_activity_name)]
    esm_location = 'CA-QC'
    end_use_tech_list = list(model[model.Output.isin(flows_list)].Name.unique())

    if flows_list == ['ELECTRICITY_EHV', 'ELECTRICITY_HV']:
        # the high and extra high voltage electricity are merged in the LCI database,
        # thus we remove transformations between these two levels of voltage
        end_use_tech_list.remove('TRAFO_HE')
        end_use_tech_list.remove('TRAFO_EH')
    elif flows_list == ['NG_HP', 'NG_EHP']:
        # the high and extra high pressure natural gas are merged in the LCI database,
        # thus we remove transformations between these two levels of pressure
        end_use_tech_list.remove('NG_EXP_EH')
        end_use_tech_list.remove('NG_EXP_EH_COGEN')
        end_use_tech_list.remove('NG_COMP_HE')
    elif flows_list == ['H2_MP', 'H2_EHP']:
        # the medium and extra high pressure hydrogen are merged in the LCI database,
        # thus we remove transformations between these two levels of pressure
        end_use_tech_list.remove('H2_COMP_HE')
        end_use_tech_list.remove('H2_EXP_EH')
        end_use_tech_list.remove('H2_EXP_EH_COGEN')
    else:
        pass

    total_amount = 0  # initialize the total amount of production

    for tech in end_use_tech_list:
        amount = esm_results[esm_results.Name == tech].Production.iloc[0]
        total_amount += amount

    exchanges = []

    for tech in end_use_tech_list:
        print(tech)
        amount = esm_results[esm_results.Name == tech].Production.iloc[0]
        if amount == 0:
            pass
        else:
            activity_name = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')].Activity.iloc[0]
            activity_prod = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')].Product.iloc[0]
            activity_database = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')].Database.iloc[0]
            activity_location = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')].Location.iloc[0]
            activity_unit = db_dict_name[activity_name, activity_prod, activity_location, activity_database]['unit']
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

    try:  # Check if the original activity is in the database for the location under study
        original_activity = db_dict_name[
            original_activity_name, original_activity_prod, esm_location, original_activity_database
        ]

    except KeyError:  # If not, we create it for the location under study
        new_code = random_code()

        similar_act = [a for a in wurst.get_many(db, *[
            wurst.equals('name', original_activity_name),
            wurst.equals('reference product', original_activity_prod),
            wurst.equals('database', original_activity_database)
        ])][0]

        CPC_cat = dict(similar_act['classifications'])['CPC']
        original_activity_unit = similar_act['unit']

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
        new_activity = {
            'database': original_activity_database,
            'name': original_activity_name,
            'location': esm_location,
            'unit': original_activity_unit,
            'reference product': original_activity_prod,
            'code': new_code,
            'classifications': CPC_cat,
            'comment': f'activity derived from EnergyScope results in the layers {flows_list} for {esm_location}.',
            'parameters': {},
            'categories': None,
            'exchanges': exchanges
        }

        db.append(new_activity)

        # Plugging the new activity in the database
        activities_of_esm_region = [a for a in wurst.get_many(db, *[wurst.equals('location', esm_location)])]
        for act in activities_of_esm_region:
            technosphere_flows = get_technosphere_flows(act)
            for exc in technosphere_flows:
                if (exc['name'] == original_activity_name) and (exc['product'] == original_activity_prod):
                    exc['code'] = new_activity['code']
                    exc['location'] = esm_location
                else:
                    pass

    else:  # If it is in the database, we update it (i.e., replace all technosphere exchanges)
        for exc in original_activity['exchanges']:  # Remove all technosphere exchanges
            if exc['type'] == 'technosphere':
                original_activity['exchanges'].remove(exc)
            else:
                pass
        original_activity['exchanges'] += exchanges  # Add the new technosphere exchanges

    return db