import copy
from .database import Database, Dataset
import wurst


def _regionalize_activity_foreground(
        self,
        act: dict,
) -> dict:
    """
    Regionalize a foreground activity according to the user ranking of locations

    :param act: activity to regionalize
    :return: the regionalized activity
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    esm_location = self.esm_location
    accepted_locations = self.accepted_locations
    spatialized_biosphere_db = self.spatialized_biosphere_db
    main_database = self.main_database
    spatialized_database = self.spatialized_database
    db_dict_code = main_database.db_as_dict_code
    db_dict_name = main_database.db_as_dict_name

    if act['location'] in accepted_locations:
        new_act = copy.deepcopy(act)

    # Imports and exports are special cases for which we do not regionalize
    elif act['name'].split(',')[0] in self.import_export_list:
        new_act = copy.deepcopy(act)

    else:
        new_act = copy.deepcopy(act)
        new_act_name = new_act['name']
        new_act_product = new_act['reference product']
        new_act['comment'] = f'This LCI dataset has been adapted to {esm_location}. ' + new_act.get('comment', '')
        new_act['location'] = esm_location
        prod_flow = Dataset(new_act).get_production_flow()
        prod_flow['location'] = esm_location

        technosphere_flows = Dataset(new_act).get_technosphere_flows()

        # for each technosphere flow, we choose the best possible location according to the user ranking
        for flow in technosphere_flows:
            techno_act = db_dict_code[(flow['database'], flow['code'])]
            techno_act_name = techno_act['name']
            techno_act_product = techno_act['reference product']
            techno_act_location = techno_act['location']
            techno_act_database = techno_act['database']

            # if a flow is the same as the product (e.g., markets) we do not change the location to avoid
            # infinite loops
            if (techno_act_name == new_act_name) & (techno_act_product == new_act_product):
                continue

            if techno_act_location in accepted_locations:
                continue

            new_location = self._change_location_activity(
                product=techno_act_product,
                activity=techno_act_name,
                location=techno_act_location,
                database=techno_act_database,
                technosphere_or_biosphere_db=main_database,
                activity_type='technosphere',
            )  # best possible location according to the user ranking

            if new_location != techno_act_location:  # if the best location is different from the initial location
                new_techno_act = db_dict_name[
                    (techno_act_name, techno_act_product, new_location, techno_act_database)
                ]
                flow['database'] = new_techno_act['database']
                flow['code'] = new_techno_act['code']
                flow['location'] = new_techno_act['location']
                flow['input'] = (new_techno_act['database'], new_techno_act['code'])
                flow['comment'] = f'Changed from {techno_act_location} to {new_location}' + flow.get('comment', '')

        if spatialized_database:
            spatialized_biosphere_db_name = [i for i in spatialized_biosphere_db.db_as_list][0]['database']
            biosphere_flows = Dataset(new_act).get_biosphere_flows()

            for flow in biosphere_flows:
                if flow['database'] == spatialized_biosphere_db_name:  # if the biosphere flow is regionalized
                    flow_name = flow['name']
                    current_loc = None
                    generic_name = None

                    for i in range(1, flow_name.count(', ') + 1):
                        try_current_loc = ', '.join(flow_name.split(', ')[-i:])
                        try_generic_name = ', '.join(flow_name.split(', ')[:-i])
                        if try_current_loc in self.locations_list:
                            current_loc = try_current_loc
                            generic_name = try_generic_name
                            break

                    if current_loc is None:
                        self.logger.warning(
                            f'Could not regionalize biosphere flow {flow_name} in {flow["database"]}. '
                            f'No location found in the name.'
                        )
                        continue

                    if current_loc in accepted_locations:
                        continue

                    new_location = self._change_location_activity(
                        activity=generic_name,
                        categories=flow['categories'],
                        location=current_loc,
                        database=flow['database'],
                        technosphere_or_biosphere_db=spatialized_biosphere_db,
                        activity_type='biosphere',
                    )  # best possible location according to the user ranking

                    if new_location != current_loc:  # if the best location is different from the initial location
                        new_flow_name = f"{generic_name}, {new_location}"
                        new_biosphere_act = spatialized_biosphere_db.list_to_dict(
                            key='name',
                            database_type='biosphere'
                        )[(new_flow_name, flow['categories'], flow['database'])]

                        flow['name'] = new_biosphere_act['name']
                        flow['code'] = new_biosphere_act['code']
                        flow['input'] = (new_biosphere_act['database'], new_biosphere_act['code'])
                        flow['comment'] = f'Changed from {current_loc} to {new_location}' + flow.get('comment', '')

    return new_act


def _change_location_activity(
        self,
        activity: str,
        location: str,
        database: str,
        technosphere_or_biosphere_db: Database,
        esm_tech_name: str = None,
        product: str = None,
        categories: tuple = None,
        activity_type: str = 'technosphere',
) -> str:
    """
    Changes the location of a process given a ranking of preferred locations

    :param esm_tech_name: name of the technology or resource in the ESM
    :param product: name of the product in the LCI database for technosphere flows
    :param activity: name of the activity in the LCI database
    :param location: initial location of the process
    :param database: name of the database in the Brightway project
    :param activity_type: type of activity, can be 'technosphere' or 'biosphere'
    :param categories: name of the categories in the LCI database (for biosphere flows only)
    :param technosphere_or_biosphere_db: technosphere or biosphere LCI database, depending on activity_type
    :return: the highest available location within the ranking, or the initial location is any of the listed
        locations is available
    """

    locations = []
    locations_ranking = self.locations_ranking

    if activity_type == 'technosphere':
        if (product, activity, database) in self.best_loc_in_ranking.keys():
            return self.best_loc_in_ranking[(product, activity, database)]
        act_filter = [
            wurst.searching.equals("name", activity),
            wurst.searching.equals("reference product", product),
            wurst.searching.equals("database", database)
        ]
    elif activity_type == 'biosphere':
        if (activity, categories, database) in self.best_loc_in_ranking.keys():
            return self.best_loc_in_ranking[(activity, categories, database)]
        act_filter = [
            wurst.searching.startswith("name", activity),
            wurst.searching.equals("categories", categories),
            wurst.searching.equals("database", database)
        ]
    else:
        raise ValueError("Activity type must be either 'technosphere' or 'biosphere'")

    ds = [a for a in wurst.searching.get_many(technosphere_or_biosphere_db.db_as_list, *act_filter)]

    for act in ds:
        if activity_type == 'technosphere':
            locations.append(act['location'])
        elif activity_type == 'biosphere':
            locations.append(act['name'].split(', ')[-1])

    # Imports and exports are special cases for which we keep the initial location
    if esm_tech_name in self.import_export_list:
        if activity_type == 'technosphere':
            self.best_loc_in_ranking[(product, activity, database)] = location
        elif activity_type == 'biosphere':
            self.best_loc_in_ranking[(activity, categories, database)] = location
        return location

    # special case where there is only one location
    elif len(locations) == 1:
        if activity_type == 'technosphere':
            self.best_loc_in_ranking[(product, activity, database)] = locations[0]
        elif activity_type == 'biosphere':
            self.best_loc_in_ranking[(activity, categories, database)] = locations[0]
        return locations[0]

    # normal case where we follow the ranking
    else:
        for loc in locations_ranking:
            if loc in locations:
                if activity_type == 'technosphere':
                    self.best_loc_in_ranking[(product, activity, database)] = loc
                elif activity_type == 'biosphere':
                    self.best_loc_in_ranking[(activity, categories, database)] = loc
                return loc

    if activity_type == 'technosphere':
        self.logger.warning(f'No location found in your ranking for ({product}, {activity}) in the database {database}. '
                            f'Have to keep the initial location: {location}')
        self.best_loc_in_ranking[(product, activity, database)] = location
    elif activity_type == 'biosphere':
        self.logger.warning(f'No location found in your ranking for ({activity}, {categories}) in the database {database}. '
                            f'Have to keep the initial location: {location}')
        self.best_loc_in_ranking[(activity, categories, database)] = location

    return location


def change_location_mapping_file(self) -> None:
    """
    Changes the location of a process given a mapping file

    :return: None
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    mapping = self.mapping
    main_database = self.main_database
    main_database_as_list = main_database.db_as_list

    if 'Location' not in mapping.columns:
        for i in range(len(mapping)):
            activity_name = mapping.Activity.iloc[i]
            product_name = mapping.Product.iloc[i]
            location = [a for a in wurst.get_many(main_database_as_list, *[
                wurst.equals('name', activity_name),
                wurst.equals('reference product', product_name)
            ])][0]['location']  # picks one location randomly
            mapping.at[i, 'Location'] = location

    mapping['Location'] = mapping.apply(lambda row: self._change_location_activity(
        esm_tech_name=row['Name'],
        product=row['Product'],
        activity=row['Activity'],
        location=row['Location'],
        database=row['Database'],
        technosphere_or_biosphere_db=main_database,
    ), axis=1)

    self.mapping = mapping
