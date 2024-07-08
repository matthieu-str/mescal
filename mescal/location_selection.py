import pandas as pd
import wurst


def change_location_activity(esm_tech_name: str or None, product: str, activity: str, location: str, database: str,
                             locations_ranking: list[str], db: list[dict], esm_region: str) -> str:
    """
    Changes the location of a process given a ranking of preferred locations

    :param esm_tech_name: name of the technology or resource in the energy system model
    :param product: name of the product in the LCI database
    :param activity: name of the activity in the LCI database
    :param location: initial location of the process
    :param database: name of the database in the brightway project
    :param locations_ranking: list of preferred locations
    :param db: dictionary of the LCI database
    :param esm_region: name of the modeled region in the energy system model
    :return: the highest available location within the ranking, or the initial location is any of the list's
        locations is available
    """

    locations = []

    act_filter = [
        wurst.searching.equals("name", activity),
        wurst.searching.equals("reference product", product),
        wurst.searching.equals("database", database)
    ]

    ds = [a for a in wurst.searching.get_many(db, *act_filter)]

    for act in ds:
        locations.append(act['location'])

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

    print(f'No location found in your ranking for {esm_tech_name} - {product} - {activity} - {database}')
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
