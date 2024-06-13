import pandas as pd
import copy
from mescal.utils import *


def create_complementary_database(df_mapping, premise_db, name_complement_db):
    """
    Relink the technologies to the premise database
    :param df_mapping: (pandas dataframe) dataframe with the mapping of the technologies and resources
    :param premise_db: (list of dict) premise database
    :param name_complement_db: (str) name of the complementary database
    :return: (pandas dataframe) dataframe with the mapping of the technologies and resources linked to the premise
    database
    """

    name_premise_db = premise_db[0]['database']
    tech_premise = pd.DataFrame(columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database'])
    complement_premise = []

    base_db = concatenate_databases(list(df_mapping.Database.unique()))
    base_db_dict_name = database_list_to_dict(base_db, 'name')

    premise_db_dict_name = database_list_to_dict(premise_db, 'name')
    premise_db_dict_code = database_list_to_dict(premise_db, 'code')

    for i in range(len(df_mapping)):

        esm_tech_name = df_mapping.Name.iloc[i]
        act_type = df_mapping.Type.iloc[i]
        product = df_mapping.Product.iloc[i]
        activity = df_mapping.Activity.iloc[i]
        region = df_mapping.Location.iloc[i]
        database = df_mapping.Database.iloc[i]

        # Remove year
        new_activity = activity.replace(', 2020', '')

        # Cars and SUVs inventories
        new_product = product.replace('-TEMP', '')
        new_activity = new_activity.replace('-TEMP', '')
        new_product = new_product.replace('EURO-6d', 'EURO-6ab')
        new_activity = new_activity.replace('EURO-6d', 'EURO-6ab')

        try:
            premise_db_dict_name[(new_activity, new_product, region, name_premise_db)]

        except KeyError:
            new_product = new_product[0].lower() + new_product[1:]
            new_activity = new_activity[0].lower() + new_activity[1:]

            try:
                premise_db_dict_name[(new_activity, new_product, region, name_premise_db)]

            except KeyError:
                print(f"No inventory in the premise database for {esm_tech_name, act_type}")
                complement_premise.append((esm_tech_name, act_type))
                tech_premise.loc[i] = [esm_tech_name, act_type, product, activity, region, database]

            else:
                tech_premise.loc[i] = [esm_tech_name, act_type, new_product, new_activity, region, name_premise_db]

        else:
            tech_premise.loc[i] = [esm_tech_name, act_type, new_product, new_activity, region, name_premise_db]

    for i in range(len(complement_premise)):
        esm_tech_name = complement_premise[i][0]
        act_type = complement_premise[i][1]
        activity = tech_premise[(tech_premise.Name == esm_tech_name)
                                & (tech_premise.Type == act_type)].Activity.iloc[0]
        product = tech_premise[(tech_premise.Name == esm_tech_name)
                               & (tech_premise.Type == act_type)].Product.iloc[0]
        region = tech_premise[(tech_premise.Name == esm_tech_name)
                              & (tech_premise.Type == act_type)].Location.iloc[0]
        database = tech_premise[(tech_premise.Name == esm_tech_name)
                                & (tech_premise.Type == act_type)].Database.iloc[0]

        act = base_db_dict_name[(activity, product, region, database)]

        try:
            premise_db_dict_name[(activity, product, region, name_complement_db)]
        except KeyError:
            new_act = copy.deepcopy(act)
            new_code = random_code()
            new_act['database'] = name_complement_db
            new_act['code'] = new_code
            prod_flow = get_production_flow(new_act)
            prod_flow['code'] = new_code
            prod_flow['database'] = name_complement_db
            premise_db.append(new_act)
            premise_db_dict_name[
                (new_act['name'], new_act['reference product'], new_act['location'], new_act['database'])] = new_act
            premise_db_dict_code[(new_act['database'], new_act['code'])] = new_act

    unlinked_activities = [i for i in premise_db if i['database'] == name_complement_db]
    while len(unlinked_activities) > 0:
        unlinked_activities, premise_db = relink(name_complement_db, base_db, premise_db)

    complement_db = [i for i in premise_db if i['database'] == name_complement_db]
    write_wurst_database_to_brightway(complement_db, name_complement_db)

    tech_premise.reset_index(inplace=True, drop=True)
    tech_premise_adjusted = pd.DataFrame(columns=tech_premise.columns)

    for i in range(len(tech_premise)):
        esm_tech_name = tech_premise.Name.iloc[i]
        act_type = tech_premise.Type.iloc[i]
        product = tech_premise.Product.iloc[i]
        activity = tech_premise.Activity.iloc[i]
        region = tech_premise.Location.iloc[i]
        database = tech_premise.Database.iloc[i]

        if database == name_premise_db:
            tech_premise_adjusted.loc[i] = [esm_tech_name, act_type, product, activity, region, database]
        else:
            tech_premise_adjusted.loc[i] = [esm_tech_name, act_type, product, activity, region, name_complement_db]

    return tech_premise_adjusted


def relink(name_complement_db, base_db, premise_db):
    """
    :param name_complement_db: (str) name of the complementary database
    :param base_db: (list of dictionaries) list of activities in the base database
    :param premise_db: (list of dictionaries) list of activities in the premise database
    :return: (list of dict) list of unlinked flows, (list of dict) updated premise database
    """

    name_premise_db = premise_db[0]['database']
    unlinked_activities = []
    complement_database = [i for i in premise_db if i['database'] == name_complement_db]
    premise_db_dict_name = database_list_to_dict(premise_db, 'name')
    base_db_dict_name = database_list_to_dict(base_db, 'name')

    for act in complement_database:
        technosphere_flows = get_technosphere_flows(act)

        for flow in technosphere_flows:
            database = flow['database']
            if (database == name_premise_db) | (database == name_complement_db):
                pass
            else:
                activity = flow['name']
                if 'reference product' in flow.keys():
                    product = flow['reference product']
                elif 'product' in flow.keys():
                    product = flow['product']
                else:
                    raise ValueError('No reference product found')
                region = flow['location']

                try:
                    act_db = premise_db_dict_name[(activity, product, region, name_complement_db)]
                except KeyError:
                    try:
                        act_premise = premise_db_dict_name[(activity, product, region, name_premise_db)]
                    except KeyError:
                        try:
                            act_premise_lc = premise_db_dict_name[(
                                activity[0].lower() + activity[1:], product[0].lower() + product[1:], region,
                                name_premise_db)]
                        except KeyError:
                            act_comp = base_db_dict_name[(activity, product, region, database)]
                            new_act = copy.deepcopy(act_comp)
                            new_code = random_code()
                            new_act['database'] = name_complement_db
                            new_act['code'] = new_code
                            prod_flow = get_production_flow(new_act)
                            prod_flow['code'] = new_code
                            prod_flow['database'] = name_complement_db
                            flow['code'] = new_code
                            flow['database'] = name_complement_db
                            unlinked_activities.append(new_act)
                            premise_db.append(new_act)
                            premise_db_dict_name[(new_act['name'], new_act['reference product'], new_act['location'],
                                                  new_act['database'])] = new_act
                        else:
                            code = act_premise_lc['code']
                            flow['code'] = code
                            flow['database'] = name_premise_db
                    else:
                        code = act_premise['code']
                        flow['code'] = code
                        flow['database'] = name_premise_db
                else:
                    code = act_db['code']
                    flow['code'] = code
                    flow['database'] = name_complement_db

    return unlinked_activities, premise_db
