import wurst
from .utils import write_wurst_database_to_brightway


def add_CPC_category(db, product, CPC_category, type):
    """
    Add a CPC category to a set of activities in a wurst database
    :param db: (list of dict) LCI database
    :param product: (str) name of the product
    :param CPC_category: (str) CPC category
    :param type: (str) type of search: 'equals' or 'contains'
    :return: (list of dict) updated LCI database
    """
    if type == 'equals':
        act_list = [a for a in wurst.get_many(db, *[wurst.searching.equals("reference product", product)])]
    elif type == 'contains':
        act_list = [a for a in wurst.get_many(db, *[wurst.searching.contains("reference product", product)])]
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


def create_new_database_with_CPC_categories(db, new_db_name, mapping_product_to_CPC):
    """
    Create a new database with additional CPC categories
    :param db: (list of dict) LCI database
    :param new_db_name: (str) name of the new database
    :param mapping_product_to_CPC: (pd.DataFrame) mapping between products and CPC categories
    :return: None
    """
    for i in range(len(mapping_product_to_CPC)):
        product = mapping_product_to_CPC.Product.iloc[i]
        CPC_category = mapping_product_to_CPC.CPC.iloc[i]
        type = mapping_product_to_CPC.Type.iloc[i]
        db = add_CPC_category(db, product, CPC_category, type)

    write_wurst_database_to_brightway(db, new_db_name)
