import pandas as pd
from .utils import ecoinvent_unit_convention
from .filesystem_constants import DATA_DIR


def load_change_report_annex(v_from: str, v_to: str) -> pd.DataFrame:
    """
    Load the change report annex between two versions of the ecoinvent database

    :param v_from: initial version of the ecoinvent database
    :param v_to: next version of the ecoinvent database
    :return: change report annex as a pandas DataFrame
    """
    df = pd.read_excel(io=DATA_DIR / "ecoinvent_change_reports" / f"Change Report Annex v{v_from} - v{v_to}.xlsx",
                       sheet_name="Qualitative Changes",
                       usecols=[
                           f'Reference Product - {v_from}',
                           f'Reference Product Unit - {v_from}',
                           f'Activity Name - {v_from}',
                           f'Geography - {v_from}',
                           f'Reference Product - {v_to}',
                           f'Reference Product Unit - {v_to}',
                           f'Activity Name - {v_to}',
                           f'Geography - {v_to}',
                           f'Dataset in version {v_from[:3]} has been deleted'
                       ])

    df.rename(columns={
        f'Activity Name - {v_from}': 'Activity Name',
        f'Geography - {v_from}': 'Geography',
        f'Reference Product - {v_from}': 'Reference Product',
        f'Reference Product Unit - {v_from}': 'Unit',
        f'Activity Name - {v_to}': 'Activity Name - new',
        f'Geography - {v_to}': 'Geography - new',
        f'Reference Product - {v_to}': 'Reference Product - new',
        f'Reference Product Unit - {v_to}': 'Unit - new',
        f'Dataset in version {v_from[:3]} has been deleted' :'Deleted'
    }, inplace=True)

    df['Version from'] = v_from
    df['Version to'] = v_to

    return df


def concatenate_change_reports(v_from: str, v_to: str) -> pd.DataFrame:
    """
    Concatenate change reports annexes of the ecoinvent database

    :param v_from: initial version of the ecoinvent database
    :param v_to: final version of the ecoinvent database
    :return: concatenated change report annex as a pandas DataFrame
    """
    ecoinvent_versions = ['3.8', '3.9', '3.9.1', '3.10']
    change_reports = []
    i = 0
    while v_from != ecoinvent_versions[i]:
        i += 1
    while v_to != ecoinvent_versions[i]:
        change_reports.append(load_change_report_annex(ecoinvent_versions[i], ecoinvent_versions[i + 1]))
        i += 1
    return pd.concat(change_reports)


def handle_multi_processes_ecoinvent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle multi-processes activities in the ecoinvent change report annex

    :param df: dataframe of the concatenated change report annex
    :return: updated dataframe with multi-processes activities separated
    """
    updated_df = pd.DataFrame(data=[], columns=df.columns)
    for i in range(len(df)):
        if ';' in str(df['Reference Product'].iloc[i]):

            products = df['Reference Product'].iloc[i].split(';\n')
            units = df['Unit'].iloc[i].split(';\n')

            if str(df['Reference Product - new'].iloc[i]) == 'nan':
                new_products = ['nan'] * len(products)
            else:
                new_products = df['Reference Product - new'].iloc[i].split(';\n')
            if str(df['Unit - new'].iloc[i]) == 'nan':
                new_units = ['nan'] * len(units)
            else:
                new_units = df['Unit - new'].iloc[i].split(';\n')

            for product, unit, new_product, new_unit in zip(products, units, new_products, new_units):
                updated_df.loc[len(updated_df)] = [df['Activity Name'].iloc[i], df['Geography'].iloc[i], product, unit,
                                                   df['Activity Name - new'].iloc[i], df['Geography - new'].iloc[i],
                                                   new_product, new_unit, df['Deleted'].iloc[i],
                                                   df['Version from'].iloc[i], df['Version to'].iloc[i]]
        else:
            updated_df.loc[len(updated_df)] = df.iloc[i].tolist()

    return updated_df


def load_concatenated_ecoinvent_change_report(v_from: str, v_to: str) -> pd.DataFrame:
    """
    Load the concatenated change report between two versions of the ecoinvent database

    :param v_from: initial version of the ecoinvent database
    :param v_to: next version of the ecoinvent database
    :return: concatenated change report as a pandas DataFrame
    """
    df = concatenate_change_reports(v_from, v_to)
    df = handle_multi_processes_ecoinvent(df)
    df = df.reset_index(drop=True)

    df_glo = df[(df['Geography'] == 'GLO') & (df['Geography - new'] == 'GLO')]  # global activities

    for i in range(len(df_glo)):
        # Add new row for the same activity but with RoW as location to fill missing locations in the change report
        new_row = df_glo.iloc[i].tolist()
        new_row[1], new_row[5] = 'RoW', 'RoW'
        df.loc[len(df)] = new_row

    # Only keep rows with changes in reference product, activity name or geography
    df = df.drop(df[
                     (df['Reference Product - new'] == df['Reference Product'])
                     & (df['Activity Name - new'] == df['Activity Name'])
                     & (df['Geography - new'] == df['Geography'])
                     ].index)

    return df


def update_mapping_file(mapping: pd.DataFrame, change_report: pd.DataFrame, unit_to_change: list = None) \
        -> tuple[pd.DataFrame, int, [tuple[str, str], tuple[str, str, str], str, str]]:
    """
    Update the mapping file with the concatenated change report

    :param mapping: mapping between the LCI datasets and the ESM technologies
    :param change_report: concatenated change report between two versions of the ecoinvent database
    :param unit_to_change: list of tuples in case a unit change has been detected
    :return: updated mapping, number of changes, list of tuples with unit changes
    """
    changed_activities = [list(e) for e in {tuple(item) for item in change_report[
        ['Reference Product', 'Activity Name', 'Geography']].values.tolist()}]

    updated_mapping = pd.DataFrame(data=[], columns=mapping.columns)
    counter = 0

    if unit_to_change is None:
        unit_to_change = []

    for i in range(len(mapping)):

        activity_name = mapping['Activity'].iloc[i]
        activity_prod = mapping['Product'].iloc[i]
        activity_geo = mapping['Location'].iloc[i]
        tech_name = mapping['Name'].iloc[i]
        tech_type = mapping['Type'].iloc[i]
        database = mapping['Database'].iloc[i]

        # REMIND and IMAGE regions are not in the change report
        if activity_geo in ['CAZ', 'CHA', 'NEU', 'EUR', 'IND', 'JPN', 'LAM', 'MEA', 'OAS', 'REF', 'SSA', 'USA',
                            'RSAM', 'RCAM', 'INDO', 'RSAF', 'CEU', 'SAF', 'INDIA', 'BRA', 'STAN', 'WAF', 'CHN', 'NAF',
                            'UKR', 'RSAS', 'RUS', 'SEAS', 'KOR', 'JAP', 'EAF', 'TUR', 'CAN', 'MEX', 'WEU']:
            activity_geo = 'RoW'

        if [activity_prod, activity_name, activity_geo] in changed_activities:
            counter += 1
            activity_name_new, activity_prod_new, activity_geo_new, unit, unit_new, deleted = change_report[
                (change_report['Reference Product'] == activity_prod)
                & (change_report['Activity Name'] == activity_name)
                & (change_report['Geography'] == activity_geo)
                ][['Activity Name - new', 'Reference Product - new', 'Geography - new', 'Unit', 'Unit - new',
                   'Deleted']].iloc[0]

            # Switch to ecoinvent database standard unit convention
            unit = ecoinvent_unit_convention(unit)
            unit_new = ecoinvent_unit_convention(unit_new)

            if unit != unit_new:
                print(f"WARNING: unit changed for {activity_prod} - {activity_name} - {activity_geo}")
                unit_to_change.append(
                    [(tech_name, tech_type), (activity_prod, activity_name, activity_geo), unit, unit_new])

            if (str(activity_name) == 'nan') & (deleted == 1):
                raise ValueError(
                    f"Activity {activity_prod} - {activity_name} - {activity_geo} has been deleted in the last "
                    f"ecoinvent version and should be replaced.")

            else:
                updated_mapping.loc[i] = [tech_name, tech_type, activity_prod_new, activity_name_new, activity_geo_new,
                                          database]

                print(tech_name, tech_type)
                print(f"Old: {activity_prod} - {activity_name} - {activity_geo}")
                print(f"New: {activity_prod_new} - {activity_name_new} - {activity_geo_new}")

        else:
            updated_mapping.loc[i] = mapping.iloc[i]

    return updated_mapping, counter, unit_to_change


def change_database_name_in_mapping_file(row: pd.Series, version_from: str, version_to: str,
                                         name_complementary_db: str = None) -> pd.Series:
    """
    Change the name of the database in the mapping file

    :param row: row of the mapping file
    :param version_from: initial version of the ecoinvent database
    :param version_to: target version of the ecoinvent database
    :param name_complementary_db: name of the complementary database
    :return: updated row
    """
    if row.Database == name_complementary_db:  # if it is the complementary database
        # Carculator truck databases
        if 'urban delivery' in row.Activity:
            row.Database = 'urban delivery_truck'
        elif 'regional delivery' in row.Activity:
            row.Database = 'regional delivery_truck'
        elif 'long haul' in row.Activity:
            row.Database = 'long haul_truck'
        else:
            raise ValueError(f"{row.Name} is in the complementary database and its database could not be updated.")
    else:
        row.Database = row.Database.replace(version_from, version_to)
    return row


def change_ecoinvent_version_mapping(mapping: pd.DataFrame, v_from: str, v_to: str, name_complementary_db: str = None) \
        -> tuple[pd.DataFrame, [tuple[str, str], tuple[str, str, str], str, str]]:
    """
    Change the version of the ecoinvent database in the mapping file

    :param mapping: mapping between the LCI datasets and the ESM technologies
    :param v_from: initial version of the ecoinvent database
    :param v_to: target version of the ecoinvent database
    :param name_complementary_db: name of the complementary database
    :return: updated mapping, list of tuples with unit changes
    """

    change_report = load_concatenated_ecoinvent_change_report(v_from, v_to)
    updated_mapping, counter, unit_to_change = update_mapping_file(mapping, change_report)

    while counter > 0:
        updated_mapping, counter, unit_to_change = update_mapping_file(updated_mapping, change_report, unit_to_change)

    updated_mapping = updated_mapping.apply(
        lambda row: change_database_name_in_mapping_file(row, v_from, v_to, name_complementary_db),
        axis=1
    )

    return updated_mapping, unit_to_change


def update_unit_conversion_file(unit_conversion: pd.DataFrame, unit_changes: list, new_unit_conversion_factors: dict) \
        -> pd.DataFrame:
    """
    Adapt the unit conversion file according to the possible unit changes in the mapping file

    :param unit_conversion: file with unit conversion factors
    :param unit_changes: list of tuples with unit changes
    :param new_unit_conversion_factors: dictionary with new unit conversion factors
    :return: updated unit conversion file
    """
    for i in range(len(unit_changes)):
        unit_esm, unit_lca = unit_conversion[
            (unit_conversion.Name == unit_changes[i][0][0])
            & (unit_conversion.Type == unit_changes[i][0][1])
            ][['ESM', 'LCA']].values[0]

        if unit_lca != unit_changes[i][2]:
            raise ValueError(f'LCA unit for {unit_changes[i][0][0]} - {unit_changes[i][0][1]} '
                             f'is not the same as the one in the mapping file. {unit_lca} != {unit_changes[i][2]}')
        else:
            if unit_changes[i][0] in new_unit_conversion_factors.keys():
                new_value = new_unit_conversion_factors[unit_changes[i][0]]
            else:
                raise ValueError(f"Missing new unit conversion factor for {unit_changes[i][0]}")

            # delete current row
            unit_conversion = unit_conversion.drop(unit_conversion[
                                                       (unit_conversion.Name == unit_changes[i][0][0])
                                                       & (unit_conversion.Type == unit_changes[i][0][1])
                                                       ].index)

            # add new row
            unit_conversion.loc[len(unit_conversion)] = [unit_changes[i][0][0], unit_changes[i][0][1], new_value,
                                                         unit_changes[i][3], unit_esm]

    return unit_conversion
