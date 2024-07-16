import os
import random
import string
import wurst
import bw2data as bd
import pandas as pd
from .caching import cache_database, load_db
from .filesystem_constants import DIR_DATABASE_CACHE


def load_extract_db(db_name: str, create_pickle: bool = False) -> list[dict]:
    """
    Load or extract a database

    :param db_name: name of the database
    :param create_pickle: if True, create a pickle file to store the database
    :return: list of activities of the LCI database
    """
    if db_name not in bd.databases:
        raise ValueError(f"{db_name} is not a registered database")
    elif os.path.isfile(DIR_DATABASE_CACHE / f'{db_name}.pickle'):
        db = load_db(db_name)
    else:
        db = wurst.extract_brightway2_databases(db_name, add_identifiers=True)
        if create_pickle:
            cache_database(db, db_name)
    return db


def load_multiple_databases(database_list: list[str], create_pickle: bool = False) -> list[dict]:
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
            if (dep_db_name not in database_list) & ('biosphere' not in dep_db_name):
                db += load_extract_db(dep_db_name, create_pickle)
                database_list.append(dep_db_name)
            else:
                pass
    return db


def merge_databases(database_list: list[str], new_db_name: str, main_ecoinvent_db_name: str,
                    old_main_db_names: list[str] or None = None, output: str = 'return') -> list[dict] or None:
    """
    Merge multiple LCI databases in one database. The list of databases should contain one main database (e.g., an
    ecoinvent or premise database) towards which all other databases will be relinked.

    :param database_list: list of LCI databases to merge
    :param new_db_name: name of the new merged database
    :param main_ecoinvent_db_name: name of the main database, e.g., ecoinvent or premise database
    :param old_main_db_names: other main databases that are not in the list of databases, thus the list of databases
        will be unlinked from those
    :param output: 'return' to return the merged database, 'write' to write it, 'both' to do both
    :return: the newly created database or None if output is 'write'
    """
    merged_db = []
    main_ecoinvent_db = load_extract_db(main_ecoinvent_db_name)

    for db_name in database_list:
        if db_name != main_ecoinvent_db_name:
            db = load_extract_db(db_name)
            for old_db_name in old_main_db_names:
                db = relink_database(db=db,
                                     name_database_unlink=old_db_name,
                                     name_database_relink=main_ecoinvent_db_name,
                                     output='return',
                                     db_relink=main_ecoinvent_db)
        else:
            db = main_ecoinvent_db
        merged_db += db

    # Verification that the merged database has no dependencies apart from biosphere databases
    dependencies = list(set([a['exchanges'][i]['database'] for a in merged_db for i in range(len(a['exchanges']))]))
    for dep_db_name in dependencies:
        if 'biosphere' in dep_db_name:
            pass
        elif dep_db_name not in database_list:
            raise ValueError(f"Database {dep_db_name} is not in the list of databases to merge")

    if output == 'write':
        write_wurst_database_to_brightway(merged_db, new_db_name)
    elif output == 'return':
        return merged_db
    elif output == 'both':
        write_wurst_database_to_brightway(merged_db, new_db_name)
        return merged_db
    else:
        raise ValueError('The output argument must be either "return", "write" or "both"')


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
            if 'input' in exc.keys():
                if exc['input'][0] in old_dbs_name:
                    exc['input'] = (new_db_name, exc['input'][1])
            if 'output' in exc.keys():
                if exc['output'][0] in old_dbs_name:
                    exc['output'] = (new_db_name, exc['output'][1])
            else:
                pass
    return db


def relink_database(db: list[dict], name_database_unlink: str, name_database_relink: str,
                    name_new_db: str = None, output: str = 'write', db_relink: list[dict] or None = None) \
        -> None or list[dict]:
    """
    Relink a database based on activity codes and write/return it

    :param db: list of activities of the LCI database
    :param name_database_unlink: name of the database to unlink
    :param name_database_relink: name of the database to relink
    :param name_new_db: name of the new database, if None, the original database is overwritten
    :param output: 'write' to write the new database, 'return' to return it, or 'both' to do both
    :param db_relink: list of activities of the database to relink to. If None, the database will be loaded using
        name_database_relink
    :return: the new database or None if output is 'write'
    """
    if db_relink is None:
        db_relink = load_extract_db(name_database_relink)
    db_relink_dict_code = database_list_to_dict(db_relink, 'code')

    if name_new_db is None:
        name_new_db = db[0]['database']
    for act in db:
        for exc in act['exchanges']:
            if exc['database'] == name_database_unlink:
                if (name_database_relink, exc['code']) in db_relink_dict_code:
                    exc['database'] = name_database_relink
                else:
                    raise ValueError(f"Flow {exc['code']} not found in database {name_database_relink}")
            if 'input' in exc.keys():
                if exc['input'][0] == name_database_unlink:
                    if (name_database_relink, exc['input'][1]) in db_relink_dict_code:
                        exc['input'] = (name_database_relink, exc['input'][1])
                    else:
                        raise ValueError(f"Flow {exc['input'][1]} not found in database {name_database_relink}")
    if output == 'write':
        write_wurst_database_to_brightway(db, name_new_db)
    elif output == 'return':
        return db
    elif output == 'both':
        write_wurst_database_to_brightway(db, name_new_db)
        return db
    else:
        raise ValueError('Output must be either "write", "return" or "both"')


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
    old_db_names = list(set([act['database'] for act in db]))
    if (len(old_db_names) > 1) | (old_db_names[0] != db_name):
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
        'kg/h': 'kilogram per hour',
        'm2': 'square meter',
        'm3': 'cubic meter',
        'MJ': 'megajoule',
        'kWh': 'kilowatt hour',
        'kW': 'kilowatt',
        'h': 'hour',
        'km': 'kilometer',
        'pkm': 'person kilometer',
        'pkm/h': 'person kilometer per hour',
        'tkm': 'ton kilometer',
        'tkm/h': 'ton kilometer per hour',
        'u': 'unit',
    }
    if unit in unit_dict:
        return unit_dict[unit]
    elif unit in [unit_dict[u] for u in unit_dict]:
        return unit
    else:
        raise ValueError(f"Unmapped unit {unit}")


def premise_changing_names(activity_name: str, activity_prod: str, activity_loc: str, name_premise_db,
                           premise_db_dict_name: dict, premise_changes: pd.DataFrame = None) \
        -> tuple[str, str, str]:
    """
    Returns the updated name, product and location in case some changes have occurred in premise

    :param activity_name: name of the LCI dataset
    :param activity_prod: product of the LCI dataset
    :param activity_loc: location of the LCI dataset
    :param name_premise_db: name of the premise database
    :param premise_db_dict_name: dictionary of the database with (name, product, location, database) as key
    :param premise_changes: file of the premise name changes impacting the mapping
    :return: the updated name, product and location of the LCI dataset
    """

    if (activity_name, activity_prod, activity_loc, name_premise_db) in premise_db_dict_name:
        return activity_name, activity_prod, activity_loc
    elif (activity_name, activity_prod, "RoW", name_premise_db) in premise_db_dict_name:
        return activity_name, activity_prod, "RoW"
    elif premise_changes is None:
        return activity_name, activity_prod, activity_loc
    else:
        try:
            activity_name_new, activity_prod_new, activity_loc_new = premise_changes[
                (premise_changes['Activity - old'] == activity_name)
                & (premise_changes['Product - old'] == activity_prod)
                & (premise_changes['Location - old'] == activity_loc)
                ][['Activity - new', 'Product - new', 'Location - new']].values[0]
        except IndexError:  # the LCI dataset is not in the premise database
            return activity_name, activity_prod, activity_loc
        else:
            return activity_name_new, activity_prod_new, activity_loc_new


def change_year_in_name(row: pd.Series, year_from: int, year_to: int) -> pd.Series:
    """
    Change the year in the name of the activity and database

    :param row: row of the mapping file
    :param year_from: year of the original mapping file
    :param year_to: year of the new mapping file
    :return: updated mapping row
    """
    row['Activity'] = row['Activity'].replace(str(year_from), str(year_to))
    row['Database'] = row['Database'].replace(str(year_from), str(year_to))
    return row


def change_mapping_year(mapping: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    """
    Change the year in the name of the activities and databases in the mapping file

    :param mapping: mapping file between the LCI database and the ESM database
    :param year_from: year of the original mapping file
    :param year_to: year of the new mapping file
    :return: updated mapping file
    """
    if year_from == year_to:
        print(f'The mapping file is already for the year {year_to}')
        return mapping
    else:
        mapping = mapping.apply(lambda row: change_year_in_name(row, year_from, year_to), axis=1)
        return mapping


def test_mapping_file(mapping: pd.DataFrame, db: list[dict]) -> list[tuple[str, str, str, str]]:
    """
    Test if the mapping file is correctly linked to the database

    :param mapping: mapping file between the LCI database and the ESM database
    :param db: LCI database
    :return: the list missing flows if any
    """
    missing_flows = []
    db_dict_name = database_list_to_dict(db, 'name')
    for i in range(len(mapping)):
        act_name = mapping.Activity.iloc[i]
        act_prod = mapping.Product.iloc[i]
        act_loc = mapping.Location.iloc[i]
        act_database = mapping.Database.iloc[i]
        if (act_name, act_prod, act_loc, act_database) in db_dict_name:
            pass
        else:
            missing_flows.append((act_name, act_prod, act_loc, act_database))
    if len(missing_flows) > 0:
        print(f'Some flows could be not found in the database: {missing_flows}')
    else:
        print('Mapping successfully linked to the database')
    return missing_flows
