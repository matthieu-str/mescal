import wurst
import pandas as pd
from .filesystem_constants import DATA_DIR


def add_product_or_activity_CPC_category(self, name: str, CPC_category: str, search_type: str, key: str) -> None:
    """
    Add a CPC category to a set of activities in a LCI database

    :param name: name or part of the name of the product or activity
    :param CPC_category: CPC category to add
    :param search_type: type of search, it can be 'equals' (i.e., exact name matching) or 'contains' (i.e., name is
        in the activity name or product name)
    :param key: searching key, it can be 'name' or 'reference product'
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
                pass  # if all activities already have a CPC category, we do not overwrite it


def add_CPC_categories(
        self,
        new_db_name: str = None,
        mapping_product_to_CPC: pd.DataFrame or str = 'default',
        write: bool = False,
) -> None:
    """
    Add CPC categories to a database, and write it as a new database if asked

    :param new_db_name: name of the new database (only required if write is True)
    :param mapping_product_to_CPC: mapping between products and CPC categories, can be a pandas DataFrame or the
        path towards the csv file
    :param write: if True, write the new database to Brightway
    :return: None
    """

    if type(mapping_product_to_CPC) is str:
        if mapping_product_to_CPC == 'default':
            mapping_product_to_CPC = pd.read_csv(DATA_DIR / 'mapping_product_to_CPC.csv')
        else:
            mapping_product_to_CPC = pd.read_csv(mapping_product_to_CPC)
    elif type(mapping_product_to_CPC) is pd.DataFrame:
        pass
    else:
        raise ValueError('Mapping must be either a pandas DataFrame or a path towards a csv file')

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
        self.add_product_or_activity_CPC_category(name, CPC_category, search_type, key)

    if write:
        if new_db_name is None:
            raise ValueError('The "new_db_name" argument must be provided if "write" is True')
        self.write_to_brightway(new_db_name)
