import random
import string
import pandas as pd


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


def expand_impact_category_levels(
        df: pd.DataFrame,
        impact_category_col: str = 'Impact_category',
) -> pd.DataFrame:
    """
    Expand the impact category levels into separate columns

    :param df: dataframe with impact category column
    :param impact_category_col: name of the impact category column
    :return: the dataframe with expanded impact category levels
    """

    max_len = df[impact_category_col].dropna().apply(
        lambda x: len(x) if isinstance(x, (tuple, list)) else 0
    ).max()

    expanded = pd.DataFrame(
        df[impact_category_col].apply(
            lambda x: list(x) + [None] * (max_len - len(x)) if isinstance(x, (tuple, list)) else [None] * max_len
        ).tolist(),
        index=df.index
    )

    expanded.columns = [f'{impact_category_col} (level {i})' for i in range(max_len)]

    return pd.concat([df, expanded], axis=1)


def _short_name_ds_type(ds_type: str) -> str:
    """
    Returns the short name of the LCI dataset type

    :param ds_type: type of LCI dataset
    :return: short name of the LCI dataset type
    """
    if ds_type == 'Construction':
        return 'constr'
    elif ds_type == 'Decommission':
        return 'decom'
    elif ds_type == 'Operation':
        return 'op'
    elif ds_type == 'Resource':
        return 'res'
    else:
        raise ValueError(f"Unknown technology type: {ds_type}")