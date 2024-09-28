import os
import random
import string
import wurst
import bw2data as bd
import pandas as pd
from .caching import cache_database, load_db
from .filesystem_constants import DIR_DATABASE_CACHE


class Dataset:

    def __init__(self, act: dict) -> None:
        """
        Initialize the dataset

        :param act: LCI dataset
        """
        self.act = act
        self.product = act['reference product']
        self.activity = act['name']
        self.location = act['location']
        self.database = act['database']

    def get_technosphere_flows(self) -> list[dict]:
        """
        Get the technosphere flows of an activity

        :return: list of technosphere flows
        """
        flows = []
        for exc in self.act['exchanges']:
            if exc['type'] == 'technosphere':
                flows.append(exc)
        return flows

    def get_biosphere_flows(self) -> list[dict]:
        """
        Get the biosphere flows of an activity

        :return: list of biosphere flows
        """
        flows = []
        for exc in self.act['exchanges']:
            if exc['type'] == 'biosphere':
                flows.append(exc)
        return flows

    def get_production_flow(self) -> dict or None:
        """
        Get the production flow of an activity

        :return: production flow or None if it does not exist
        """
        for exc in self.act['exchanges']:
            if exc['type'] == 'production':
                return exc
        return None

    def get_downstream_consumers(self, db: list[dict]) -> list[dict]:
        """
        Get the downstream consumers of an activity

        :param db: list of activities of the LCI database
        :return: list of downstream consumers
        """
        act_code = self.act['code']
        consumers = []
        for a in db:
            ds = Dataset(a)
            for exc in ds.get_technosphere_flows():
                if exc['code'] == act_code:
                    consumers.append(a)
                    continue
        return consumers


class Database:

    def __init__(
            self,
            db_names: str | list[str] = None,
            db_as_list: list[dict] = None,
            create_pickle: bool = False
    ) -> None:
        """
        Initialize the database

        :param db_names: Name of the LCI database(s). Should be a string for a single database or a list of strings for
            several databases.
        :param db_as_list: List of dictionaries of the LCI database.
        :param create_pickle: if True, create a pickle file to store the database. Only used if db_names is provided.
        """
        if db_as_list is not None:
            self.db_as_list = db_as_list
            self.db_names = list(set([i['database'] for i in db_as_list]))
        elif db_names is not None:
            self.db_names = db_names
            self.db_as_list = self.load(create_pickle)
        else:
            raise ValueError('Database name or list of dictionaries must be provided')

    @property
    def db_as_dict_name(self) -> dict:
        return self.__class__.list_to_dict(self, key='name')

    @property
    def db_as_dict_code(self) -> dict:
        return self.__class__.list_to_dict(self, key='code')

    @db_as_dict_name.setter
    def db_as_dict_name(self, db_dict_name: dict) -> None:
        self.db_names = list(set([i[3] for i in db_dict_name.keys()]))
        self.db_as_list = list(db_dict_name.values())

    @db_as_dict_code.setter
    def db_as_dict_code(self, db_dict_code: dict) -> None:
        self.db_names = list(set([i[0] for i in db_dict_code.keys()]))
        self.db_as_list = list(db_dict_code.values())

    def load(self, create_pickle: bool = False) -> list[dict]:
        """
        Load or extract a single database

        :param create_pickle: if True, create a pickle file to store the database
        :return: list of dictionaries of the LCI database
        """
        if isinstance(self.db_names, list):
            db = self.load_multiple(create_pickle)
        elif isinstance(self.db_names, str):
            if self.db_names not in bd.databases:
                raise ValueError(f"{self.db_names} is not a registered database")
            elif os.path.isfile(DIR_DATABASE_CACHE / f'{self.db_names}.pickle'):
                db = load_db(self.db_names)
            else:
                db = wurst.extract_brightway2_databases(self.db_names, add_identifiers=True)
                if create_pickle:
                    cache_database(db, self.db_names)
        else:
            raise ValueError('Database name must be a string or a list of strings')
        return db

    def load_multiple(self, create_pickle: bool = False) -> list[dict]:
        """
        Concatenates databases in a list of dictionaries (including dependencies)

        :param create_pickle: if True, create a pickle file to store the database
        :return: list of dictionaries of the concatenated databases
        """
        db = []
        for name in self.db_names:
            db += self.__class__(name).load(create_pickle)
            dependencies = list(set([a['exchanges'][i]['database'] for a in db for i in range(len(a['exchanges']))]))
            for dep_db_name in dependencies:
                if (dep_db_name not in self.db_names) & ('biosphere' not in dep_db_name):
                    db += self.__class__(dep_db_name).load(create_pickle)
                    self.db_names.append(dep_db_name)
                else:
                    pass
        return db

    def merge(
            self,
            main_ecoinvent_db_name: str,
            new_db_name: str = None,
            old_main_db_names: list[str] or None = None,
            write: bool = False,
            check_duplicates: bool = False
    ) -> list[dict] or None:
        """
        Merge multiple LCI databases in one database. The list of databases should contain one main database (e.g., an
        ecoinvent or premise database) towards which all other databases will be relinked.

        :param main_ecoinvent_db_name: name of the main database, e.g., ecoinvent or premise database
        :param new_db_name: name of the new merged database. Only used if output is 'write' or 'both'.
        :param old_main_db_names: other main databases that are not in the list of databases, thus the list of databases
            will be unlinked from those
        :param write: if True, write the new database to Brightway
        :param check_duplicates: if True, check for duplicates in terms of (product, name, location) and remove them
            from exchanges and from the database
        :return: None
        """

        main_ecoinvent = Database(main_ecoinvent_db_name)
        main_ecoinvent_db = main_ecoinvent.db_as_list

        for db_name in self.db_names:
            db = Database(db_name)
            if db_name != main_ecoinvent_db_name:
                for old_db_name in old_main_db_names:
                    db.relink(
                        name_database_unlink=old_db_name,
                        name_database_relink=main_ecoinvent_db_name,
                    )
            else:
                db.db_as_list = main_ecoinvent_db
            self.db_as_list += db.db_as_list

        # Checking for duplicates in terms of (product, name, location)
        if check_duplicates:
            db_dict_name = main_ecoinvent.db_as_dict_name
            db_dict_name_count = {k[:3]: 0 for k in db_dict_name.keys()}

            for act in self.db_as_list:  # counting the number of occurrences of each activity
                db_dict_name_count[(act['name'], act['reference product'], act['location'])] += 1

            for k, v in db_dict_name_count.items():
                if v > 1:  # duplicates are removed from exchanges and from the database

                    ref_act = None
                    for db_name in self.db_names:
                        key = (k[0], k[1], k[2], db_name)
                        if key in db_dict_name:
                            ref_act = db_dict_name[key]  # one of the duplicates is kept

                    for act in self.db_as_list:
                        if (
                                (act['name'] == ref_act['name'])
                                & (act['reference product'] == ref_act['reference product'])
                                & (act['location'] == ref_act['location'])
                        ):  # if we find a duplicate
                            if act['code'] != ref_act['code']:  # it is removed if its code is not the same as the reference
                                self.db_as_list.remove(act)
                        else:  # for other activities, we update the exchanges

                            for exc in Dataset(act).get_technosphere_flows():
                                if (
                                        (exc['name'] == ref_act['name'])
                                        & (exc['product'] == ref_act['reference product'])
                                        & (exc['location'] == ref_act['location'])
                                ):  # if we find a duplicate in the exchanges, its code is updated with the reference
                                    exc['code'] = ref_act['code']
                                    exc['database'] = ref_act['database']
                                    exc['input'] = (ref_act['database'], ref_act['code'])

        # Verification that the merged database has no dependencies apart from biosphere databases
        dependencies = list(set([a['exchanges'][i]['database'] for a in self.db_as_list for i in range(len(a['exchanges']))]))
        for dep_db_name in dependencies:
            if 'biosphere' in dep_db_name:
                pass
            elif main_ecoinvent_db_name in dep_db_name:
                pass
            elif dep_db_name not in self.db_names:
                raise ValueError(f"Database {dep_db_name} is not in the list of databases to merge")

        if write:
            if new_db_name is None:
                raise ValueError('The "new_db_name" argument must be provided if "write" is True')
            self.write_to_brightway(new_db_name)

    def list_to_dict(
            self,
            key: str,
            database_type: str = 'technosphere'
    ) -> dict:
        """
        Converts a list of dictionaries into a dictionary with the (database, code) tuple or the (name, product, location,
        database) tuple as key.

        :param key: 'code' or 'name'
        :param database_type: 'technosphere' or 'biosphere'
        :return: LCI database as a dictionary
        """
        if key == 'code':
            db_as_dict = {(a['database'], a['code']): a for a in self.db_as_list}
        elif key == 'name':
            if database_type == 'technosphere':
                db_as_dict = {(a['name'], a['reference product'], a['location'], a['database']): a for a in self.db_as_list}
            elif database_type == 'biosphere':
                db_as_dict = {(a['name'], a['categories'], a['database']): a for a in self.db_as_list}
            else:
                raise ValueError('Database type must be either "technosphere" or "biosphere"')
        else:
            raise ValueError('Key must be either "code" or "name"')

        return db_as_dict

    def wurst_to_brightway(self) -> None:
        """
        Adjust the database to the Brightway2 format

        :return: None
        """

        for act in self.db_as_list:

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

    def change_name(self, new_db_name: str) -> None:
        """
        Change the name of the database

        :param new_db_name: new name of the database
        :return: None
        """

        old_dbs_name = list(set([act['database'] for act in self.db_as_list]))
        for act in self.db_as_list:
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

    def relink(
            self,
            name_database_unlink: str,
            name_database_relink: str,
            name_new_db: str = None,
            write: bool = False,
    ) -> None:
        """
        Relink a database based on activity codes and write it

        :param name_database_unlink: name of the database to unlink
        :param name_database_relink: name of the database to relink
        :param name_new_db: name of the new database, if None, the original database is overwritten (if write is True)
        :param write: if True, write the new database to Brightway
        :return: None
        """

        relink_db = Database(name_database_relink)
        db_relink_dict_code = relink_db.list_to_dict('code')

        if name_new_db is None:
            name_new_db = self.db_as_list[0]['database']
        for act in self.db_as_list:
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
        if write:
            self.write_to_brightway(name_new_db)

    def write_to_brightway(self, new_db_name: str) -> None:
        """
        Write a wurst database to a Brightway2 project. This function will overwrite the database if it already exists

        :param new_db_name: name of the brightway database to be written
        :return: None
        """

        if new_db_name in list(bd.databases):  # if the database already exists, delete it
            del bd.databases[new_db_name]
        else:
            pass
        bw_database = bd.Database(new_db_name)
        bw_database.register()
        old_db_names = list(set([act['database'] for act in self.db_as_list]))
        if (len(old_db_names) > 1) | (old_db_names[0] != new_db_name):
            self.change_name(new_db_name)
        self.wurst_to_brightway()
        db = {(i['database'], i['code']): i for i in self.db_as_list}
        bw_database.write(db)

    def get_code(self, product, activity, location, database) -> str or None:
        """
        Get the code of an activity

        :param product: product of the LCI dataset
        :param activity: name of the LCI dataset
        :param location: location of the LCI dataset
        :param database: name of the LCI database
        :return: activity code in the LCI database or None if it does not exist
        """

        ds = self.db_as_dict_name[(activity, product, location, database)]

        if 'code' in ds.keys():
            return ds['code']

        else:
            return None

    def test_mapping_file(self, mapping: pd.DataFrame) -> list[tuple[str, str, str, str]]:
        """
        Test if the mapping file is correctly linked to the database

        :param mapping: mapping file between the LCI database and the ESM database
        :return: the list missing flows if any
        """
        missing_flows = []
        for i in range(len(mapping)):
            act_name = mapping.Activity.iloc[i]
            act_prod = mapping.Product.iloc[i]
            act_loc = mapping.Location.iloc[i]
            act_database = mapping.Database.iloc[i]
            if (act_name, act_prod, act_loc, act_database) in self.db_as_dict_name:
                pass
            else:
                missing_flows.append((act_name, act_prod, act_loc, act_database))
        if len(missing_flows) > 0:
            print(f'Some flows could be not found in the database: {missing_flows}')
        else:
            print('Mapping successfully linked to the database')
        return missing_flows

    def dependencies(self) -> list[str]:
        """
        Get the dependencies of the database

        :return: list of dependencies
        """
        dependencies = list(set([a['exchanges'][i]['database'] for a in self.db_as_list for i in range(len(a['exchanges']))]))
        return dependencies

def ecoinvent_unit_convention(unit: str) -> str:
    """
    Reformat unit to the ecoinvent convention

    :param unit: unit to reformat
    :return: ecoinvent unit
    """
    unit_dict = {
        'kg': 'kilogram',
        'kg*day': 'kilogram day',
        'kg/h': 'kilogram per hour',
        'm2': 'square meter',
        'm3': 'cubic meter',
        'MJ': 'megajoule',
        'kWh': 'kilowatt hour',
        'kW': 'kilowatt',
        'h': 'hour',
        'km': 'kilometer',
        'km*year': 'kilometer-year',
        'pkm': 'person kilometer',
        'person*km': 'person kilometer',
        'pkm/h': 'person kilometer per hour',
        'tkm': 'ton kilometer',
        'metric ton*km': 'ton kilometer',
        'tkm/h': 'ton kilometer per hour',
        'u': 'unit',
    }
    if unit in unit_dict:
        return unit_dict[unit]
    elif unit in [unit_dict[u] for u in unit_dict]:
        return unit
    else:
        raise ValueError(f"Unmapped unit {unit}")


def premise_changing_names(
        activity_name: str,
        activity_prod: str,
        activity_loc: str,
        name_premise_db,
        premise_db_dict_name: dict,
        premise_changes: pd.DataFrame = None
) -> tuple[str, str, str]:
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


def random_code() -> str:
    """
    Create a random code

    :return: code
    """
    length = 32
    code_rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return code_rand
