import pandas as pd
import wurst


def change_location_activity(activity: str, location: str, database: str, locations_ranking: list[str], db: list[dict],
                             esm_region: str, esm_tech_name: str = None,  product: str = None,
                             categories: tuple = None, activity_type: str = 'technosphere') -> str:
    """
    Changes the location of a process given a ranking of preferred locations

    :param esm_tech_name: name of the technology or resource in the energy system model
    :param product: name of the product in the LCI database for technosphere flows
    :param activity: name of the activity in the LCI database
    :param location: initial location of the process
    :param database: name of the database in the brightway project
    :param locations_ranking: list of preferred locations
    :param db: dictionary of the LCI database
    :param esm_region: name of the modeled region in the energy system model
    :param activity_type: type of activity, can be 'technosphere' or 'biosphere'
    :param categories: name of the categories in the LCI database for biosphere flows
    :return: the highest available location within the ranking, or the initial location is any of the list's
        locations is available
    """

    locations = []

    if activity_type == 'technosphere':
        act_filter = [
            wurst.searching.equals("name", activity),
            wurst.searching.equals("reference product", product),
            wurst.searching.equals("database", database)
        ]
    elif activity_type == 'biosphere':
        act_filter = [
            wurst.searching.startswith("name", activity),
            wurst.searching.equals("categories", categories),
            wurst.searching.equals("database", database)
        ]
    else:
        raise ValueError("Activity type must be either 'technosphere' or 'biosphere'")

    ds = [a for a in wurst.searching.get_many(db, *act_filter)]

    for act in ds:
        if activity_type == 'technosphere':
            locations.append(act['location'])
        elif activity_type == 'biosphere':
            locations.append(act['name'].split(', ')[-1])

    # special case of Quebec for electricity imports that should come from the US
    if (esm_region == 'CA-QC') & (esm_tech_name in ['ELECTRICITY_EHV',
                                                    'ELECTRICITY_HV',
                                                    'ELECTRICITY_LV',
                                                    'ELECTRICITY_MV']):
        return 'US-NPCC'

    # special case where there is only one location
    elif len(locations) == 1:
        return locations[0]

    # normal case where we follow the ranking
    else:
        for loc in locations_ranking:
            if loc in locations:
                return loc

    if activity_type == 'technosphere':
        print(f'No location found in your ranking for {product} - {activity} - {database}')
    elif activity_type == 'biosphere':
        print(f'No location found in your ranking for {activity} - {categories} - {database}')
    print(f'Have to keep the initial location: {location}')
    return location


def change_location_mapping_file(df_mapping: pd.DataFrame, locations_ranking: list[str],
                                 database: list[dict], esm_region: str) -> pd.DataFrame:
    """
    Changes the location of a process given a mapping file

    :param df_mapping: dataframe with the mapping of the technologies and resources
    :param locations_ranking: list of preferred locations by order of preference
    :param database: dictionary of the LCI database
    :param esm_region: name of the modeled region in the energy system model
    :return: dataframe with the mapping of the technologies and resources with the new location
    """

    if 'Location' not in df_mapping.columns:
        for i in range(len(df_mapping)):
            activity_name = df_mapping.Activity.iloc[i]
            product_name = df_mapping.Product.iloc[i]
            location = [a for a in wurst.get_many(database, *[
                wurst.equals('name', activity_name),
                wurst.equals('reference product', product_name)
            ])][0]['location']  # picks one location randomly
            df_mapping.at[i, 'Location'] = location

    df_mapping['Location'] = df_mapping.apply(lambda row: change_location_activity(
        esm_tech_name=row['Name'],
        product=row['Product'],
        activity=row['Activity'],
        location=row['Location'],
        database=row['Database'],
        locations_ranking=locations_ranking,
        db=database,
        esm_region=esm_region
    ), axis=1)

    return df_mapping
