import os
import wurst
import copy
import bw2data as bd
import pandas as pd
import logging
from .caching import cache_database, load_db
from .utils import premise_changing_names, random_code
from .filesystem_constants import DIR_DATABASE_CACHE


class Dataset:
    """
    Class to perform basic operations on LCI datasets (as dictionaries)
    """

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

    def __repr__(self) -> str:
        return f"Dataset({self.product}, {self.activity}, {self.location}, {self.database})"

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
    """
    Class to perform basic operations on LCI databases. Databases can be formulated as a list of dictionaries or as a
    dictionary with the (database, code) tuple or the (name, product, location, database) tuple as key.
    """
    def __init__(
            self,
            db_names: str | list[str] = None,
            db_as_list: list[dict] = None,
            create_pickle: bool = False
    ):
        """
        Initialize the database

        :param db_names: Name of the LCI database(s). Should be a string for a single database or a list of strings for
            several databases.
        :param db_as_list: List of dictionaries of the LCI database.
        :param create_pickle: if True, create a pickle file to store the database. Only used if db_names is provided.
        """
        # set up logging tool
        self.logger = logging.getLogger('Database')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        if db_as_list is not None:
            self.db_as_list = db_as_list
            self.db_names = list(set([i['database'] for i in db_as_list]))
        elif db_names is not None:
            self.db_names = db_names
            self.db_as_list = self.load(create_pickle)
        else:
            raise ValueError('Database name or list of dictionaries must be provided')

    def __repr__(self) -> str:
        return f"Database({self.db_names}) contains {len(self.db_as_list)} activities"

    @property
    def db_as_dict_name(self) -> dict:
        return self.list_to_dict(key='name')

    @property
    def db_as_dict_code(self) -> dict:
        return self.list_to_dict(key='code')

    @db_as_dict_name.setter
    def db_as_dict_name(self, db_dict_name: dict) -> None:
        self.db_names = list(set([i[3] for i in db_dict_name.keys()]))
        self.db_as_list = list(db_dict_name.values())

    @db_as_dict_code.setter
    def db_as_dict_code(self, db_dict_code: dict) -> None:
        self.db_names = list(set([i[0] for i in db_dict_code.keys()]))
        self.db_as_list = list(db_dict_code.values())

    def __add__(self, other):
        return Database(db_as_list=self.db_as_list + other.db_as_list)

    def __sub__(self, other):
        return Database(db_as_list=[i for i in self.db_as_list if i not in other.db_as_list])

    def __len__(self):
        return len(self.db_as_list)

    from .CPC import (
        add_product_or_activity_CPC_category,
        add_CPC_categories,
        add_CPC_categories_based_on_existing_activities,
        save_mapping_between_products_and_CPC_categories,
    )

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
                self.logger.info(f"Loaded {self.db_names} from pickle!")
            else:
                db = wurst.extract_brightway2_databases(self.db_names, add_identifiers=True)
                self.logger.info(f"Loaded {self.db_names} from brightway!")
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
        db = Database(db_as_list=[])
        loaded_databases = set()
        db_names_copy = copy.deepcopy(self.db_names)
        for name in db_names_copy:
            if name not in loaded_databases:
                db += Database(db_names=name, create_pickle=create_pickle)
                loaded_databases.add(name)
                dependencies = list(set([a['exchanges'][i]['database']
                                         for a in db.db_as_list
                                         for i in range(len(a['exchanges']))]))
                for dep_db_name in dependencies:
                    if (dep_db_name not in self.db_names) & ('biosphere' not in dep_db_name):
                        if dep_db_name not in loaded_databases:
                            db += Database(db_names=dep_db_name, create_pickle=create_pickle)
                            loaded_databases.add(dep_db_name)
                        self.db_names.append(dep_db_name)
        return db.db_as_list

    def merge(
            self,
            main_ecoinvent_db_name: str,
            new_db_name: str = None,
            old_main_db_names: list[str] or None = None,
            write: bool = False,
            check_duplicates: bool = False
    ) -> None:
        """
        Merge multiple LCI databases in one database. The list of databases should contain one main database (e.g., an
        ecoinvent or premise database) towards which all other databases will be relinked. If you suspect that there
        might be duplicated LCI datasets over the different databases, you can set check_duplicates to True to remove
        them.

        :param main_ecoinvent_db_name: name of the main database, e.g., ecoinvent or premise database
        :param new_db_name: name of the new merged database (only required if write is True)
        :param old_main_db_names: other main databases that are not in the list of databases, thus the list of databases
            will be unlinked from those
        :param write: if True, write the new database to Brightway
        :param check_duplicates: if True, check for duplicates in terms of (product, name, location) and remove them
            from exchanges and from the database
        :return: None
        """

        merged_db_as_list = []  # initialize the list of dictionaries for the merged database
        main_ecoinvent = Database(db_as_list=[i for i in self.db_as_list if i['database'] == main_ecoinvent_db_name])
        main_ecoinvent_db = main_ecoinvent.db_as_list

        for db_name in self.db_names:
            db = Database(db_as_list=[i for i in self.db_as_list if i['database'] == db_name])
            if db_name != main_ecoinvent_db_name:
                for old_db_name in old_main_db_names:
                    db.relink(
                        name_database_unlink=old_db_name,
                        database_relink_as_list=main_ecoinvent_db,
                    )
            else:
                db.db_as_list = main_ecoinvent_db
            merged_db_as_list += db.db_as_list
        self.db_as_list = merged_db_as_list

        # Checking for duplicates in terms of (product, name, location)
        if check_duplicates:
            db_dict_name = self.db_as_dict_name
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
        Converts a list of dictionaries into a dictionary with the (database, code) tuple as key when key = 'code', the
        (name, product, location, database) tuple as key when key = 'name' for a technosphere database,
        or the (name, categories, database) tuple as key when key = 'name' for a biosphere database.

        :param key: can be 'code' or 'name'
        :param database_type: can be 'technosphere' or 'biosphere'
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

    def wurst_to_brightway(
            self,
            database_type: str = 'technosphere',
    ) -> None:
        """
        Adjust the database to the Brightway format

        :param database_type: type of database to be written, can be 'technosphere' or 'biosphere'
        :return: None
        """

        for act in self.db_as_list:

            if database_type == 'technosphere':  # only activities from technosphere databases have exchanges
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

            if database_type == 'technosphere':  # only activities from biosphere databases should have categories
                if "categories" in act:
                    del act["categories"]

            elif database_type == 'biosphere':
                act['type'] = 'natural resource' if act['categories'][0] == 'natural resource' else 'emission'
                if "reference product" in act:
                    del act["reference product"]
                if "location" in act:
                    del act["location"]

    def change_name(
            self,
            new_db_name: str,
            database_type: str = 'technosphere',
    ) -> None:
        """
        Change the name of the database

        :param new_db_name: new name of the database
        :param database_type: type of database to be written, can be 'technosphere' or 'biosphere'
        :return: None
        """

        old_dbs_name = list(set([act['database'] for act in self.db_as_list]))
        for act in self.db_as_list:
            act['database'] = new_db_name
            if database_type == 'technosphere':  # only activities from technosphere databases have exchanges
                for exc in act['exchanges']:
                    if exc['database'] in old_dbs_name:
                        exc['database'] = new_db_name
                    if 'input' in exc.keys():
                        if exc['input'][0] in old_dbs_name:
                            exc['input'] = (new_db_name, exc['input'][1])
                    if 'output' in exc.keys():
                        if exc['output'][0] in old_dbs_name:
                            exc['output'] = (new_db_name, exc['output'][1])
        self.db_names = new_db_name

    def relink(
            self,
            name_database_unlink: str,
            name_database_relink: str = None,
            database_relink_as_list: list[dict] = None,
            name_new_db: str = None,
            based_on: str = 'code',
            write: bool = False,
    ) -> None:
        """
        Relink a database based on activity codes and write it

        :param name_database_unlink: name of the database to unlink
        :param name_database_relink: name of the database to relink
        :param database_relink_as_list: list of dictionaries of the database to relink
        :param name_new_db: name of the new database, if None, the original database is overwritten (if write is True)
        :param based_on: can be 'code' or 'name', if 'code', the relinking is done based on the code of the activities,
            if 'name', the relinking is done based on the name, product, location and database of the activities.
        :param write: if True, write the new database to Brightway
        :return: None
        """

        if database_relink_as_list is not None:
            relink_db = Database(db_as_list=database_relink_as_list)
        elif name_database_relink is not None:
            relink_db = Database(name_database_relink)
        else:
            raise ValueError("'name_database_relink' or 'database_relink_as_list' must be provided")

        if name_database_relink is None:
            if isinstance(relink_db.db_names, str):
                name_database_relink = relink_db.db_names
            else:
                if len(relink_db.db_names) > 1:
                    raise ValueError(f'More than one database to relink: {relink_db.db_names}. '
                                     f'Please provide a unique database name.')
                else:
                    name_database_relink = relink_db.db_names[0]

        if based_on == 'code':
            db_relink_dict = relink_db.list_to_dict('code')
        elif based_on == 'name':
            db_relink_dict = relink_db.list_to_dict('name')
        else:
            raise ValueError("based_on must be either 'code' or 'name'")

        if name_new_db is None:
            name_new_db = self.db_as_list[0]['database']
        for act in self.db_as_list:
            for exc in act['exchanges']:
                if exc['database'] == name_database_unlink:
                    if based_on == 'code':
                        if (name_database_relink, exc['code']) in db_relink_dict:
                            exc['database'] = name_database_relink
                        else:
                            raise ValueError(f"Flow {exc['code']} not found in database {name_database_relink}")
                    elif based_on == 'name':
                        if (exc['name'], exc['product'], exc['location'], name_database_relink) in db_relink_dict:
                            new_exc = db_relink_dict[(exc['name'], exc['product'], exc['location'], name_database_relink)]
                            exc['database'] = name_database_relink
                            exc['code'] = new_exc['code']
                        else:
                            raise ValueError(f"Flow ({exc['product']}-{exc['name']}-{exc['location']}) not found "
                                             f"in database {name_database_relink}")

                if 'input' in exc.keys():
                    if based_on == 'code':
                        if exc['input'][0] == name_database_unlink:
                            if (name_database_relink, exc['input'][1]) in db_relink_dict:
                                exc['input'] = (name_database_relink, exc['input'][1])
                            else:
                                raise ValueError(f"Flow {exc['input'][1]} not found in database {name_database_relink}")
                    elif based_on == 'name':
                        if exc['input'][0] == name_database_unlink:
                            if (exc['name'], exc['product'], exc['location'], name_database_relink) in db_relink_dict:
                                new_exc = db_relink_dict[(exc['name'], exc['product'], exc['location'], name_database_relink)]
                                exc['input'] = (name_database_relink, new_exc['code'])
                            else:
                                raise ValueError(f"Flow ({exc['product']}-{exc['name']}-{exc['location']}) not found "
                                                 f"in database {name_database_relink}")
        if write:
            self.write_to_brightway(name_new_db)

    def write_to_brightway(
            self,
            new_db_name: str,
            database_type: str = 'technosphere',
            overwrite: bool = True,
    ) -> None:
        """
        Write a LCI database to a Brightway project. This function will overwrite the database if it already exists.

        :param new_db_name: name of the brightway database to be written
        :param database_type: type of database to be written, can be 'technosphere' or 'biosphere'
        :param overwrite: if True, overwrite the database if it already exists
        :return: None
        """

        if database_type not in ['technosphere', 'biosphere']:
            raise ValueError('Database type must be either "technosphere" or "biosphere"')

        if new_db_name in list(bd.databases):  # if the database already exists
            if overwrite:
                del bd.databases[new_db_name]
                self.logger.info(f"Previous {new_db_name} will be overwritten!")
            else:
                self.logger.info(f"{new_db_name} already exists in Brightway. To overwrite it, set 'overwrite' to "
                                 f"True or delete it manually")
                return
        else:
            pass
        bw_database = bd.Database(new_db_name)
        bw_database.register()
        old_db_names = list(set([act['database'] for act in self.db_as_list]))
        if (len(old_db_names) > 1) | (old_db_names[0] != new_db_name):
            self.change_name(new_db_name, database_type)
        self.wurst_to_brightway(database_type)
        db = {(i['database'], i['code']): i for i in self.db_as_list}
        bw_database.write(db)
        self.logger.info(f"{new_db_name} written to Brightway!")

    def delete(self) -> None:
        """
        Delete a database from Brightway

        :return: None
        """
        if len(self.db_names) > 1:
            raise ValueError('Only one database can be deleted at a time')
        else:
            if self.db_names[0] in list(bd.databases):
                del bd.databases[self.db_names[0]]
            else:
                raise ValueError(f"{self.db_names[0]} is not a registered database")

    def get_code(
            self,
            product: str,
            activity: str,
            location: str,
            database: str
    ) -> str or None:
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
        db_dict_name = self.db_as_dict_name
        missing_flows = []
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
            self.logger.warning(f'Some flows could not be found in the database: {missing_flows}')
        else:
            self.logger.info('Mapping successfully linked to the database')
        return missing_flows

    def dependencies(self) -> list[str]:
        """
        Get the dependencies of the database

        :return: list of dependencies
        """
        dependencies = list(set([a['exchanges'][i]['database'] for a in self.db_as_list for i in range(len(a['exchanges']))]))
        return dependencies

    def create_complementary_database(
            self,
            df_mapping: pd.DataFrame,
            main_db_name: str = None,
            complement_db_name: str = None,
            premise_changes: pd.DataFrame = None,
            write_database: bool = True,
    ) -> pd.DataFrame:
        """
        Create a new database containing all activities that tare not provided in your main database

        :param df_mapping: dataframe with the mapping of the technologies and resources
        :param main_db_name: name of the main LCI database
        :param complement_db_name: name of the complementary LCI database
        :param premise_changes: file of the changes in names, products, locations, in premise regarding the mapping
        :param write_database: if True, write the complementary database to Brightway
        :return: dataframe with the mapping of the technologies and resources linked to the premise
            database
        """

        if (isinstance(self.db_names, str)) | (isinstance(self.db_names, list) & (len(self.db_names) == 1)):
            premise_db = self
            premise_db_list = premise_db.db_as_list
            premise_db_dict_name = premise_db.db_as_dict_name
            premise_db_dict_code = premise_db.db_as_dict_code
            name_premise_db = premise_db_list[0]['database']
            base_db = Database(db_names=list(df_mapping.Database.unique()))
            base_db_dict_name = base_db.db_as_dict_name

        elif isinstance(self.db_names, list) & (len(self.db_names) > 1):
            premise_db = Database(db_as_list=[act for act in self.db_as_list if act['database'] == main_db_name])
            premise_db_list = premise_db.db_as_list
            premise_db_dict_name = premise_db.db_as_dict_name
            premise_db_dict_code = premise_db.db_as_dict_code
            name_premise_db = main_db_name

            unlinked = self.test_mapping_file(df_mapping)
            if len(unlinked) > 0:
                base_db = Database(db_names=list(df_mapping.Database.unique()))
                base_db_dict_name = base_db.db_as_dict_name
            else:
                base_db = self
                base_db_dict_name = base_db.db_as_dict_name

        else:
            raise ValueError('Database name or list of names must be provided')

        tech_premise = pd.DataFrame(columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database'])
        complement_premise = []

        for i in range(len(df_mapping)):

            esm_tech_name = df_mapping.Name.iloc[i]
            act_type = df_mapping.Type.iloc[i]
            product = df_mapping.Product.iloc[i]
            activity = df_mapping.Activity.iloc[i]
            region = df_mapping.Location.iloc[i]
            database = df_mapping.Database.iloc[i]

            new_activity, new_product, new_location = premise_changing_names(
                activity_name=activity,
                activity_prod=product,
                activity_loc=region,
                name_premise_db=name_premise_db,
                premise_db_dict_name=premise_db_dict_name,
                premise_changes=premise_changes,
            )

            try:
                premise_db_dict_name[(new_activity, new_product, new_location, name_premise_db)]

            except KeyError:
                new_product = new_product[0].lower() + new_product[1:]
                new_activity = new_activity[0].lower() + new_activity[1:]

                try:
                    premise_db_dict_name[(new_activity, new_product, new_location, name_premise_db)]

                except KeyError:
                    self.logger.info(f"No inventory in the premise database for {esm_tech_name, act_type}")
                    complement_premise.append((esm_tech_name, act_type))
                    tech_premise.loc[i] = [esm_tech_name, act_type, product, activity, new_location, database]

                else:
                    tech_premise.loc[i] = [esm_tech_name, act_type, new_product, new_activity, new_location,
                                           name_premise_db]

            else:
                tech_premise.loc[i] = [esm_tech_name, act_type, new_product, new_activity, new_location,
                                       name_premise_db]

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
                premise_db_dict_name[(activity, product, region, complement_db_name)]
            except KeyError:
                new_act = copy.deepcopy(act)
                ds = Dataset(new_act)
                new_code = random_code()
                new_act['database'] = complement_db_name
                new_act['code'] = new_code
                prod_flow = ds.get_production_flow()
                prod_flow['code'] = new_code
                prod_flow['database'] = complement_db_name
                premise_db_list.append(new_act)
                premise_db_dict_name[
                    (new_act['name'], new_act['reference product'], new_act['location'], new_act['database'])] = new_act
                premise_db_dict_code[(new_act['database'], new_act['code'])] = new_act

        unlinked_activities = [i for i in premise_db_list if i['database'] == complement_db_name]
        while len(unlinked_activities) > 0:
            unlinked_activities, premise_db_list = self.relink_complement_db_to_premise_db(
                name_complement_db=complement_db_name,
                base_db_list=base_db.db_as_list,
                premise_db_list=premise_db_list,
                name_premise_db=name_premise_db)

        complement_db = [i for i in premise_db_list if i['database'] == complement_db_name]
        if (len(complement_db) > 0) & (write_database == True):
            Database(db_as_list=complement_db).write_to_brightway(complement_db_name)
        else:
            self.logger.info(f"The complementary database did not have to be created")

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
                tech_premise_adjusted.loc[i] = [esm_tech_name, act_type, product, activity, region, complement_db_name]

        return tech_premise_adjusted

    @staticmethod
    def relink_complement_db_to_premise_db(
            name_complement_db: str,
            base_db_list: list[dict],
            premise_db_list: list[dict],
            name_premise_db: str
    ) -> tuple[list[dict], list[dict]]:
        """
        Relink the activities in the complementary database to the premise database

        :param name_complement_db: name of the complementary database
        :param base_db_list: list of activities in the base database
        :param premise_db_list: list of activities in the premise database
        :param name_premise_db: name of the premise database
        :return: list of unlinked flows, updated premise database
        """

        unlinked_activities = []
        complement_database = [i for i in premise_db_list if i['database'] == name_complement_db]
        premise_db_dict_name = Database(db_as_list=premise_db_list).db_as_dict_name
        base_db_dict_name = Database(db_as_list=base_db_list).db_as_dict_name

        for act in complement_database:
            ds = Dataset(act)
            technosphere_flows = ds.get_technosphere_flows()

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
                                ds = Dataset(new_act)
                                new_code = random_code()
                                new_act['database'] = name_complement_db
                                new_act['code'] = new_code
                                prod_flow = ds.get_production_flow()
                                prod_flow['code'] = new_code
                                prod_flow['database'] = name_complement_db
                                flow['code'] = new_code
                                flow['database'] = name_complement_db
                                flow['input'] = (name_complement_db, new_code)
                                unlinked_activities.append(new_act)
                                premise_db_list.append(new_act)
                                premise_db_dict_name[
                                    (new_act['name'], new_act['reference product'], new_act['location'],
                                     new_act['database'])] = new_act
                            else:
                                code = act_premise_lc['code']
                                flow['code'] = code
                                flow['database'] = name_premise_db
                                flow['input'] = (name_premise_db, code)
                        else:
                            code = act_premise['code']
                            flow['code'] = code
                            flow['database'] = name_premise_db
                            flow['input'] = (name_premise_db, code)
                    else:
                        code = act_db['code']
                        flow['code'] = code
                        flow['database'] = name_complement_db
                        flow['input'] = (name_complement_db, code)

        return unlinked_activities, premise_db_list
