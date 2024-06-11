import os
import random
import string
import wurst
import bw2data as bd
from mescal.caching import cache_database, load_db


def load_extract_db(db_name):
    """
    Load or extract a database
    :param db_name: (str) name of the database
    :return: (list of dict) dictionary of the LCI database
    """
    if db_name not in bd.databases:
        raise ValueError(f"{db_name} is not a registered database")
    elif os.path.isfile(f'export/cache/{db_name}.pickle'):
        db = load_db(db_name)
    else:
        db = wurst.extract_brightway2_databases(db_name, add_identifiers=True)
        cache_database(db, db_name)
    return db


def concatenate_databases(database_list):
    """
    Concatenates databases in a list of dictionaries (including dependencies)
    :param database_list: (list of str) list of LCI database names
    :return: (list of dict) list of dictionaries of the concatenated databases
    """
    db = []
    for db_name in database_list:
        db += load_extract_db(db_name)
        dependencies = list(set([a['exchanges'][i]['database'] for a in db for i in range(len(a['exchanges']))]))
        for dep_db_name in dependencies:
            if (dep_db_name not in database_list) & (dep_db_name != 'biosphere3'):
                db += load_extract_db(dep_db_name)
                database_list.append(dep_db_name)
            else:
                pass
    return db


def database_list_to_dict(database_list, key):
    """
    Converts a list of dictionaries into a dictionary with the (database, code) tuple or the (name, product, location,
    database) tuple as key.
    :param database_list: (list of dict) LCI database
    :param key: (str) 'code' or 'name'
    :return: (dict) LCI database as a dictionary
    """
    if key == 'code':
        db_dict = {(a['database'], a['code']): a for a in database_list}
    elif key == 'name':
        db_dict = {(a['name'], a['reference product'], a['location'], a['database']): a for a in database_list}
    else:
        raise ValueError('Key must be either "code" or "name"')

    return db_dict


def get_code(db_dict_name, product, activity, region, database):
    '''
    Get the code of an activity
    :param db_dict_name: (dict) dictionary of the LCI database with the (name, product, location, database) tuple as key
    :param product: (str) product name in the LCI database
    :param activity: (str) activity name in the LCI database
    :param region: (str) region name in the LCI database
    :return: (str) activity code in the LCI database
    '''

    ds = db_dict_name[(activity, product, region, database)]

    if 'code' in ds.keys():
        return ds['code']

    else:
        return None


def random_code():
    '''
    Create a random code
    :return: (str) code
    '''
    length = 32
    code_rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return code_rand


def get_technosphere_flows(act):
    """
    Get the technosphere flows of an activity
    :param act: (dict) activity
    :return: (list of dict) list of technosphere flows
    """
    flows = []
    for exc in act['exchanges']:
        if exc['type'] == 'technosphere':
            flows.append(exc)
    return flows


def get_biosphere_flows(act):
    """
    Get the biosphere flows of an activity
    :param act: (dict) activity
    :return: (list of dict) list of biosphere flows
    """
    flows = []
    for exc in act['exchanges']:
        if exc['type'] == 'biosphere':
            flows.append(exc)
    return flows


def get_production_flow(act):
    """
    Get the production flow of an activity
    :param act: (dict) activity
    :return: (dict) production flow
    """
    for exc in act['exchanges']:
        if exc['type'] == 'production':
            return exc
    return None


def wurst_to_brightway2_database(db):
    """
    Adjust the database to the Brightway2 format
    :param db: (list of dict) dictionary of the LCI database
    :return: (list of dict) adjusted dictionary of the LCI database
    """
    for act in db:

        # Add the input and output keys in exchanges
        for exc in act['exchanges']:
            try:
                exc['input']
            except KeyError:
                exc['input'] = (exc['database'], exc['code'])
            try:
                exc['output']
            except KeyError:
                exc['output'] = (act['database'], act['code'])

        # Restore parameters to Brightway2 format which allows for uncertainty and comments
        if "parameters" in act:
            act["parameters"] = {
                name: {"amount": amount} for name, amount in act["parameters"].items()
            }

    return db


def write_wurst_database_to_brightway(db, db_name):
    """
    Write a wurst database to a Brightway2 database. This function will overwrite the database if it already exists.
    :param db: (list of dict) dictionary of the LCI database
    :param db_name: (str) name of the Brightway2 database
    :return: None
    """
    db = wurst_to_brightway2_database(db)
    db = {(i['database'], i['code']): i for i in db}
    bw_database = bd.Database(db_name)
    if db_name in list(bd.databases):  # If the database already exists, delete it to avoid duplicates
        del bd.databases[db_name]
    else:
        pass
    bw_database.register()
    bw_database.write(db)
