import wurst
import pandas as pd
import json
from .filesystem_constants import DATA_DIR


def add_CPC_categories(
        self,
        mapping_existing_products_to_CPC: dict or str = 'default',
        mapping_new_products_to_CPC: pd.DataFrame or str = 'default',
        new_db_name: str = None,
        write: bool = False,
        overwrite_existing_CPC: bool = False,
) -> None:
    """
    Add CPC categories to a database, and write it as a new database if asked

    :param mapping_existing_products_to_CPC: mapping between existing products and CPC categories, can be a dictionary
        or the path towards a json file. If 'default', the default mapping will be used.
    :param mapping_new_products_to_CPC: mapping between new products (i.e., addition with respect to ecoinvent
        products) and CPC categories, can be a pandas DataFrame or the path towards the csv file. If 'default', the
        default mapping will be used.
    :param new_db_name: name of the new database (only required if write is True)
    :param write: if True, write the new database in the Brightway project
    :param overwrite_existing_CPC: if True, overwrite existing CPC categories in the database with the ones of the
        mapping_new_products_to_CPC mapping (default is False, i.e., do not overwrite existing CPC categories)
    :return: None
    """

    if type(mapping_existing_products_to_CPC) is str:
        if mapping_existing_products_to_CPC == 'default':
            with open(DATA_DIR / 'mapping_existing_products_to_CPC.json', 'r') as fp:
                mapping_existing_products_to_CPC = json.load(fp)
        else:
            with open(mapping_existing_products_to_CPC, 'r') as fp:
                mapping_existing_products_to_CPC = json.load(fp)
    elif type(mapping_existing_products_to_CPC) is dict:
        pass
    else:
        raise ValueError('Mapping for existing products must be either a Python dictionary or a path towards a json '
                         'file')

    if type(mapping_new_products_to_CPC) is str:
        if mapping_new_products_to_CPC == 'default':
            mapping_new_products_to_CPC = pd.read_csv(DATA_DIR / 'mapping_new_products_to_CPC.csv')
        else:
            mapping_new_products_to_CPC = pd.read_csv(mapping_new_products_to_CPC)
    elif type(mapping_new_products_to_CPC) is pd.DataFrame:
        pass
    else:
        raise ValueError('Mapping for new products must be either a pandas DataFrame or a path towards a csv file')

    self._add_CPC_categories_based_on_existing_activities(mapping_existing_products_to_CPC)

    for i in range(len(mapping_new_products_to_CPC)):
        if mapping_new_products_to_CPC.Where.iloc[i] == 'Product':
            key = 'reference product'
        elif mapping_new_products_to_CPC.Where.iloc[i] == 'Activity':
            key = 'name'
        else:
            raise ValueError('In the mapping_new_products_to_CPC.csv file, the "Where" column must be either "Product" '
                             'or "Activity"')
        name = mapping_new_products_to_CPC.Name.iloc[i]
        CPC_category = mapping_new_products_to_CPC.CPC.iloc[i]
        search_type = mapping_new_products_to_CPC["Search type"].iloc[i]
        self._add_product_or_activity_CPC_category(name, CPC_category, search_type, key, overwrite_existing_CPC)

    if write:
        if new_db_name is None:
            raise ValueError('The "new_db_name" argument must be provided if "write" is True')
        self.write_to_brightway(new_db_name)


def _add_product_or_activity_CPC_category(
        self,
        name: str,
        CPC_category: str,
        search_type: str,
        key: str,
        overwrite_existing_CPC: bool,
) -> None:
    """
    Add a CPC category to a set of activities in a LCI database

    :param name: name or part of the name of the product or activity
    :param CPC_category: CPC category to add
    :param search_type: type of search, it can be 'equals' (i.e., exact name matching) or 'contains' (i.e., name is
        in the activity name or product name)
    :param key: searching key, it can be 'name' or 'reference product'
    :param overwrite_existing_CPC: if True, overwrite existing CPC categories in the database with the ones of the mapping
    :return: None
    """
    if search_type == 'equals':
        act_list = [a for a in wurst.get_many(self.db_as_list, *[wurst.searching.equals(key, name)])]
    elif search_type == 'contains':
        act_list = [a for a in wurst.get_many(self.db_as_list, *[wurst.searching.contains(key, name)])]
    else:
        raise ValueError('Type must be either "equals" or "contains"')

    for act in act_list:
        if 'classifications' not in act.keys():
            act['classifications'] = [('CPC', CPC_category)]
        else:
            if 'CPC' not in dict(act['classifications']):
                act['classifications'] += [('CPC', CPC_category)]
            else:
                if overwrite_existing_CPC:
                    dict(act['classifications'])['CPC'] = CPC_category  # overwrite the existing CPC category
                else:
                    pass  # if the activities already has a CPC category, we do not overwrite it


def _add_CPC_categories_based_on_existing_activities(
        self,
        mapping_existing_products_to_cpc: dict,
) -> None:
    """
    Add CPC categories to a database based on existing activities

    :param mapping_existing_products_to_cpc: dictionary mapping existing products to CPC categories. The latter has been
        generated using the _save_mapping_between_products_and_CPC_categories method.
    :return: None
    """
    for ds in self.db_as_list:
        if 'classifications' in ds.keys():
            if 'CPC' in dict(ds['classifications']):
                continue  # if the activity already has a CPC category, we do not overwrite it

        product = ds['reference product']
        if product in mapping_existing_products_to_cpc.keys():
            if 'classifications' not in ds.keys():
                ds['classifications'] = [('CPC', mapping_existing_products_to_cpc[product])]
            else:
                ds['classifications'] += [('CPC', mapping_existing_products_to_cpc[product])]


def _save_mapping_between_products_and_CPC_categories(
        self,
        return_dict: bool = False,
        save_dict: bool = True,
) -> dict or None:
    """
    Crate a dictionary mapping the database products and their CPC categories and return it save it as a json file

    :param return_dict: if True, return the dictionary
    :param save_dict: if True, save the dictionary as a json file in the current directory
    :return: dictionary mapping products to CPC categories (if return_type is 'return' or 'both') or None
        (if return_type is 'save')
    """
    mapping_existing_products_to_cpc = {}

    for ds in self.db_as_list:
        if ds['reference product'] not in mapping_existing_products_to_cpc.keys():
            if 'classifications' in ds.keys():
                if 'CPC' in dict(ds['classifications']):
                    mapping_existing_products_to_cpc[ds['reference product']] = dict(ds['classifications'])['CPC']

    if save_dict:
        with open('mapping_existing_products_to_CPC.json', 'w') as fp:
            json.dump(mapping_existing_products_to_cpc, fp)
    if return_dict:
        return mapping_existing_products_to_cpc
