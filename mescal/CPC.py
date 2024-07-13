import wurst
import pandas as pd
from .utils import write_wurst_database_to_brightway
from .filesystem_constants import DATA_DIR


def add_CPC_category(db: list[dict], name: str, CPC_category: str, search_type: str, key: str) -> list[dict]:
    """
    Add a CPC category to a set of activities in a wurst database

    :param db: LCI database
    :param name: name or part of the name of the product or activity
    :param CPC_category: CPC category
    :param search_type: type of search: 'equals' or 'contains'
    :param key: key to search for the product in the activities
    :return: updated LCI database
    """
    if search_type == 'equals':
        act_list = [a for a in wurst.get_many(db, *[wurst.searching.equals(key, name)])]
    elif search_type == 'contains':
        act_list = [a for a in wurst.get_many(db, *[wurst.searching.contains(key, name)])]
    else:
        raise ValueError('Type must be either "equals" or "contains"')

    for act in act_list:
        if 'classifications' not in act.keys():
            act['classifications'] = [('CPC', CPC_category)]
        else:
            if 'CPC' not in dict(act['classifications']):
                act['classifications'] += [('CPC', CPC_category)]
            else:
                pass  # if all activities already have a CPC category, we do not overwrite it

    return db


def create_new_database_with_CPC_categories(db: list[dict], new_db_name: str,
                                            mapping_product_to_CPC: pd.DataFrame or str = 'default') -> None:
    """
    Create a new database with additional CPC categories

    :param db: LCI database
    :param new_db_name: name of the new database
    :param mapping_product_to_CPC: mapping between products and CPC categories, can be a pandas DataFrame or the path
        towards the csv file
    :return: None
    """

    if mapping_product_to_CPC == 'default':
        mapping_product_to_CPC = pd.read_csv(DATA_DIR / 'mapping_product_to_CPC.csv')
    elif type(mapping_product_to_CPC) is str:
        mapping_product_to_CPC = pd.read_csv(mapping_product_to_CPC)
    else:
        pass

    for i in range(len(mapping_product_to_CPC)):
        if mapping_product_to_CPC.Where.iloc[i] == 'Product':
            key = 'reference product'
        elif mapping_product_to_CPC.Where.iloc[i] == 'Activity':
            key = 'name'
        else:
            raise ValueError('Where must be either "Product" or "Activity"')
        name = mapping_product_to_CPC.Name.iloc[i]
        CPC_category = mapping_product_to_CPC.CPC.iloc[i]
        search_type = mapping_product_to_CPC["Search type"].iloc[i]
        db = add_CPC_category(db, name, CPC_category, search_type, key)

    write_wurst_database_to_brightway(db, new_db_name)
