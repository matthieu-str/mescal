import os
import random
import string
import wurst
import bw2data as bd
from mescal.caching import cache_database, load_db


def load_extract_db(db_name: str, create_pickle: bool = False) -> list[dict]:
    """
    Load or extract a database

    :param db_name: name of the database
    :param create_pickle: if True, create a pickle file to store the database
    :return: list of activities of the LCI database
    """
    if db_name not in bd.databases:
        raise ValueError(f"{db_name} is not a registered database")
    elif os.path.isfile(f'export/cache/{db_name}.pickle'):
        db = load_db(db_name)
    else:
        db = wurst.extract_brightway2_databases(db_name, add_identifiers=True)
        if create_pickle:
            cache_database(db, db_name)
    return db


def concatenate_databases(database_list: list[str], create_pickle: bool = False) -> list[dict]:
    """
    Concatenates databases in a list of dictionaries (including dependencies)

    :param database_list: list of LCI database names
    :param create_pickle: if True, create a pickle file to store the database
    :return: list of dictionaries of the concatenated databases
    """
    db = []
    for db_name in database_list:
        db += load_extract_db(db_name, create_pickle)
        dependencies = list(set([a['exchanges'][i]['database'] for a in db for i in range(len(a['exchanges']))]))
        for dep_db_name in dependencies:
            if (dep_db_name not in database_list) & (dep_db_name != 'biosphere3'):
                db += load_extract_db(dep_db_name, create_pickle)
                database_list.append(dep_db_name)
            else:
                pass
    return db


def database_list_to_dict(database_list: list[dict], key: str) -> dict:
    """
    Converts a list of dictionaries into a dictionary with the (database, code) tuple or the (name, product, location,
    database) tuple as key.

    :param database_list: LCI database
    :param key: 'code' or 'name'
    :return: LCI database as a dictionary
    """
    if key == 'code':
        db_dict = {(a['database'], a['code']): a for a in database_list}
    elif key == 'name':
        db_dict = {(a['name'], a['reference product'], a['location'], a['database']): a for a in database_list}
    else:
        raise ValueError('Key must be either "code" or "name"')

    return db_dict


def get_code(db_dict_name: dict, product: str, activity: str, region: str, database: str) -> str or None:
    """
    Get the code of an activity

    :param db_dict_name: dictionary of the LCI database with the (name, product, location, database) tuple as key
    :param product: product name in the LCI database
    :param activity: activity name in the LCI database
    :param region: region name in the LCI database
    :param database: name of the LCI database
    :return: activity code in the LCI database or None if it does not exist
    """

    ds = db_dict_name[(activity, product, region, database)]

    if 'code' in ds.keys():
        return ds['code']

    else:
        return None


def random_code() -> str:
    """
    Create a random code

    :return: code
    """
    length = 32
    code_rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return code_rand


def get_technosphere_flows(act: dict) -> list[dict]:
    """
    Get the technosphere flows of an activity

    :param act: dictionary of the activity
    :return: list of technosphere flows
    """
    flows = []
    for exc in act['exchanges']:
        if exc['type'] == 'technosphere':
            flows.append(exc)
    return flows


def get_biosphere_flows(act: dict) -> list[dict]:
    """
    Get the biosphere flows of an activity

    :param act: dictionary of the activity
    :return: list of biosphere flows
    """
    flows = []
    for exc in act['exchanges']:
        if exc['type'] == 'biosphere':
            flows.append(exc)
    return flows


def get_production_flow(act: dict) -> dict or None:
    """
    Get the production flow of an activity

    :param act: dictionary of the activity
    :return: production flow or None if it does not exist
    """
    for exc in act['exchanges']:
        if exc['type'] == 'production':
            return exc
    return None


def wurst_to_brightway2_database(db: list[dict]) -> list[dict]:
    """
    Adjust the database to the Brightway2 format

    :param db: list of activities of the LCI database
    :return: adjusted LCI database
    """
    for act in db:

        # Add the input and output keys in exchanges
        for exc in act['exchanges']:
            if 'input' not in exc.keys():
                exc['input'] = (exc['database'], exc['code'])
            else:  # guaranteeing consistency between input, and database and code
                exc['database'] = exc['input'][0]
                exc['code'] = exc['input'][1]
            if 'output' not in exc.keys():
                exc['output'] = (act['database'], act['code'])

        # Restore parameters to Brightway2 format which allows for uncertainty and comments
        if "parameters" in act:
            act["parameters"] = {
                name: {"amount": amount} for name, amount in act["parameters"].items()
            }

        # Correct unit names
        if act['unit'] == 'ton-kilometer':
            act['unit'] = 'ton kilometer'

        if "categories" in act:
            del act["categories"]

    return db


def change_database_name(db: list[dict], new_db_name: str) -> list[dict]:
    """
    Change the name of the database

    :param db: list of activities of the LCI database
    :param new_db_name: new name of the database
    :return: updated LCI database
    """
    old_dbs_name = list(set([act['database'] for act in db]))
    for act in db:
        act['database'] = new_db_name
        for exc in act['exchanges']:
            if exc['database'] in old_dbs_name:
                exc['database'] = new_db_name
            else:
                pass
    return db


def relink_database(db: list[dict], name_database_unlink: str, name_database_relink: str,
                    name_new_db: str = None) -> None:
    """
    Relink a database and write it

    :param db: list of activities of the LCI database
    :param name_database_unlink: name of the database to unlink
    :param name_database_relink: name of the database to relink
    :param name_new_db: name of the new database, if None, the original database is overwritten
    :return: None
    """
    if name_new_db is None:
        name_new_db = db[0]['database']
    for act in db:
        for exc in act['exchanges']:
            if exc['database'] == name_database_unlink:
                exc['database'] = name_database_relink
            else:
                pass
    write_wurst_database_to_brightway(db, name_new_db)


def write_wurst_database_to_brightway(db: list[dict], db_name: str) -> None:
    """
    Write a wurst database to a Brightway2 project. This function will overwrite the database if it already exists

    :param db: list of activities of the LCI database
    :param db_name: name of the brightway database to be written
    :return: None
    """
    if db_name in list(bd.databases):  # if the database already exists, delete it
        del bd.databases[db_name]
    else:
        pass
    bw_database = bd.Database(db_name)
    bw_database.register()
    if db[0]['database'] != db_name:
        db = change_database_name(db, db_name)
    else:
        pass
    db = wurst_to_brightway2_database(db)
    db = {(i['database'], i['code']): i for i in db}
    bw_database.write(db)


def get_downstream_consumers(act: dict, db: list[dict]) -> list[dict]:
    """
    Get the downstream consumers of an activity

    :param act: dictionary of the activity
    :param db: list of activities of the LCI database
    :return: list of downstream consumers
    """
    act_code = act['code']
    consumers = []
    for a in db:
        for exc in get_technosphere_flows(a):
            if exc['code'] == act_code:
                consumers.append(a)
                continue
    return consumers


def ecoinvent_unit_convention(unit: str) -> str:
    """
    Reformat unit to the ecoinvent convention

    :param unit: unit to reformat
    :return: ecoinvent unit
    """
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
