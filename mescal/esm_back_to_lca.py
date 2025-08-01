from .database import Dataset, Database
from .utils import random_code, ecoinvent_unit_convention
import wurst
import pandas as pd
import ast
import copy
import bw2data as bd
from tqdm import tqdm


def create_new_database_with_esm_results(
        self,
        esm_results: pd.DataFrame,
        new_end_use_types: pd.DataFrame = None,
        tech_to_remove_layers: pd.DataFrame = None,
        return_database: bool = False,
        write_database: bool = True,
        remove_background_construction_flows: bool = True,
        harmonize_efficiency_with_esm: bool = True,
        harmonize_capacity_factor_with_esm: bool = False,
        esm_results_db_name: str = None,
) -> Database | None:
    """
    Create a new database with the ESM results

    :param esm_results: results of the ESM in terms of annual production and installed capacity. It should contain the
        columns 'Name', 'Production', and 'Capacity'.
    :param tech_to_remove_layers: technologies to remove from the result LCI datasets
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :param return_database: if True, return the new database
    :param write_database: if True, write the new database in the Brightway project
    :param remove_background_construction_flows: if True, the new LCI datasets undergo the double-counting removal
        process to remove background construction flows. It should be set to True in the context of a loop between
        the ESM and LCI database, in order to not count the infrastructure impacts several times over several
        time-steps. It should be set to False if the new database is meant to be shared or used as a standalone
        database.
    :param harmonize_efficiency_with_esm: if True, apply the efficiency correction to harmonize the LCI datasets with
        the ESM assumptions
    :param harmonize_capacity_factor_with_esm: if True, apply the capacity factor correction to harmonize the LCI
        datasets with the ESM assumptions. It should be set to False if the background construction flows are removed.
    :param esm_results_db_name: name of the new database with the ESM results
    :return: database of the ESM results if return_database is True, else None
    """

    if return_database is False and write_database is False:
        raise ValueError('The new database should be either returned or written.')

    if harmonize_capacity_factor_with_esm and remove_background_construction_flows:
        raise ValueError('The harmonization of capacity factors with the ESM results cannot be performed '
                         'if the background construction flows are removed. Please set one of the two to False.')

    if self.df_flows_set_to_zero is None:
        self.df_flows_set_to_zero = pd.read_csv(f'{self.results_path_file}removed_flows_list.csv')
    if self.double_counting_removal_amount is None:
        self.double_counting_removal_amount = pd.read_csv(f'{self.results_path_file}double_counting_removal.csv')

    if 'Current_code' not in self.mapping.columns:
        self._get_original_code()
    if 'New_code' not in self.mapping.columns:
        self._get_new_code()

    # Store frequently accessed instance variables in local variables inside a method
    mapping = self.mapping
    esm_location = self.esm_location

    if tech_to_remove_layers is None:
        self.tech_to_remove_layers = pd.DataFrame(columns=['Layers', 'Technologies'])
    else:
        self.tech_to_remove_layers = tech_to_remove_layers
    if new_end_use_types is None:
        new_end_use_types = pd.DataFrame(columns=['Name', 'Search type', 'Old', 'New'])

    if esm_results_db_name is not None:
        self.esm_results_db_name = esm_results_db_name

    esm_results_db_name = self.esm_results_db_name
    flows = mapping[mapping.Type == 'Flow']

    already_done = []
    perform_d_c = []

    # Create the new LCI datasets from the ESM results
    self.logger.info("Creating new LCI datasets from the ESM results...")
    for i in tqdm(range(len(flows))):

        original_activity_prod = flows.Product.iloc[i]
        original_activity_name = flows.Activity.iloc[i]
        original_activity_database = flows.Database.iloc[i]

        if (
                original_activity_name,
                original_activity_prod,
                esm_location,
                original_activity_database
        ) in already_done:
            pass

        else:
            new_perform_d_c = self._create_or_modify_activity_from_esm_results(
                original_activity_prod=original_activity_prod,
                original_activity_name=original_activity_name,
                original_activity_database=original_activity_database,
                flows=flows,
                esm_results=esm_results,
                new_end_use_types=new_end_use_types,
            )

            already_done.append((original_activity_name,
                                 original_activity_prod,
                                 esm_location,
                                 original_activity_database))

            perform_d_c += new_perform_d_c

    # Store frequently accessed instance variables in local variables inside a method
    db_as_list = self.main_database.db_as_list

    # Double counting removal of the construction activities
    double_counting_act = pd.DataFrame(
        data=perform_d_c,
        columns=['Name', 'Product', 'Activity', 'Location', 'Database', 'Current_code']
    )
    double_counting_act.drop_duplicates(inplace=True)  # remove potential duplicates

    # Adding current code to the mapping file
    mapping['Current_code'] = mapping.apply(lambda row: self.main_database.get_code(
        product=row['Product'],
        activity=row['Activity'],
        location=row['Location'],
        database=row['Database']
    ), axis=1)

    model = self.model.pivot(index='Name', columns='Flow', values='Amount').reset_index()
    model.fillna(0, inplace=True)

    N = double_counting_act.shape[1]

    double_counting_act = pd.merge(double_counting_act, model, on='Name', how='left')
    double_counting_act['CONSTRUCTION'] = double_counting_act.shape[0] * [0]
    double_counting_act = self._add_technology_specifics(double_counting_act)

    # Injecting local variables into the instance variables
    self.mapping = mapping
    self.main_database.db_as_list = db_as_list

    if remove_background_construction_flows:
        # Double-counting removal of background construction flows
        self.logger.info("Removing double-counted construction flows...")
        flows_set_to_zero, ei_removal, activities_subject_to_double_counting = self._double_counting_removal(
            df_op=double_counting_act,
            N=N,
            ESM_inputs=['OWN_CONSTRUCTION', 'CONSTRUCTION'],
            db_type='esm results',
        )

    elif (harmonize_efficiency_with_esm or harmonize_capacity_factor_with_esm) and not remove_background_construction_flows:
        # if we do not want to remove construction flows, but we want to harmonize the database with the ESM,
        # the background processes of markets should be recorded in the ESM results database following the
        # background search algorithm of the double-counting removal procedure. However, flows are NOT removed
        # during this process.
        flows_set_to_zero, ei_removal, activities_subject_to_double_counting = self._double_counting_removal(
            df_op=double_counting_act,
            N=N,
            ESM_inputs=['OWN_CONSTRUCTION', 'CONSTRUCTION'],
            db_type='esm results wo dcr',
        )

    if harmonize_efficiency_with_esm:
        # Add ESM database to the main database (to retrieve the corrected datasets)
        if self.esm_db is not None:
            esm_db = self.esm_db
        else:
            esm_db = Database(self.esm_db_name)
        self.main_database = self.main_database + esm_db

        self.logger.info("Correcting efficiency differences between ESM and LCI datasets...")
        self._correct_esm_and_lca_efficiency_differences(db_type='esm results', write_efficiency_report=False)

        # Remove the ESM database from the main database
        self.main_database = self.main_database - esm_db

    if harmonize_capacity_factor_with_esm:
        self.logger.info("Correcting capacity factor differences between ESM and LCI datasets...")
        self._correct_esm_and_lca_capacity_factor_differences(esm_results=esm_results, write_cp_report=True)

    # Injecting local variables into the instance variables
    self.main_database.db_as_list = db_as_list

    esm_results_db = Database(
        db_as_list=[act for act in self.main_database.db_as_list if act['database'] == esm_results_db_name]
    )

    if write_database:
        esm_results_db.write_to_brightway(esm_results_db_name)
        self.logger.info("Relinking the energy flows of the ESM results database to itself...")
        self.connect_esm_results_to_database(
            create_new_db=False,
            esm_results_db_name=esm_results_db_name,
            specific_db_name=esm_results_db_name,
        )

    if return_database:
        self.main_database = self.main_database - esm_results_db
        return esm_results_db

    self.main_database = self.main_database - esm_results_db

    if write_database:
        # Modifies the written database according to specifications in tech_specifics.csv
        self._modify_written_activities(db=esm_results_db, db_type='esm results')
    else:
        self.logger.info('The techs_specifics.csv file has not been applied because the database has not been written.')


def connect_esm_results_to_database(
        self,
        create_new_db: bool = False,
        new_db_name: str = None,
        specific_db_name: str = None,
        locations: list[str] or str = None,
        update_exchanges_based_on_activity_name: bool = True,
) -> None:
    """
    Connect new LCI datasets obtained from the ESM results to the main database

    :param create_new_db: if True, create a new database connected to the ESM results database. If False, directly
        modifies the main database.
    :param new_db_name: name of the new database if create_new_db is True
    :param specific_db_name: if you want to connect another database than the main database
    :param locations: list of locations to be considered for connection with the ESM results database.
        If None, only the ESM location is considered. If 'all', all locations are considered.
    :param update_exchanges_based_on_activity_name: if True, update similar flows based on the activity and product 
        names, if False, update similar flows based on the product name only.
    :return: None (copies and/or modifies the main database)
    """

    if create_new_db is True:
        if isinstance(self.main_database.db_names, list) & (len(self.main_database.db_names) > 1):
            raise ValueError('The main database should contain only one database.')
        elif isinstance(self.main_database.db_names, list) & (len(self.main_database.db_names) == 1):
            self.main_database.db_names = self.main_database.db_names[0]

    if new_db_name is not None and create_new_db is False:
        raise ValueError('The new database name should be None if create_new_db is False.')

    if new_db_name is None and create_new_db is True:
        new_db_name = self.main_database.db_names + f'_with_esm_results_for_{self.esm_location}'

    if locations is None:
        locations = [self.esm_location]

    esm_results_db_name = self.esm_results_db_name

    # Store frequently accessed instance variables in local variables inside a method
    db_dict_name = self.main_database.db_as_dict_name
    if specific_db_name is not None:
        specific_db = Database(db_names=specific_db_name)
        db_as_list = specific_db.db_as_list
        if specific_db_name == esm_results_db_name:
            esm_results_db = specific_db
            esm_results_db_dict_name = esm_results_db.db_as_dict_name
        else:
            esm_results_db = Database(db_names=esm_results_db_name)
            esm_results_db_dict_name = esm_results_db.db_as_dict_name
    else:
        db_as_list = self.main_database.db_as_list
        esm_results_db = Database(db_names=esm_results_db_name)
        esm_results_db_dict_name = esm_results_db.db_as_dict_name
    esm_location = self.esm_location
    mapping = self.mapping

    flows = mapping[mapping.Type == 'Flow']
    already_done = []

    # Activities of the ESM region
    if locations == 'all':
        activities_of_esm_region = db_as_list
    else:
        activities_of_esm_region = [
            a for loc in locations
            for a in wurst.get_many(
                db_as_list,
                wurst.equals('location', loc)
            )
        ]

    # Plugging the new activity in the database
    for i in tqdm(range(len(flows))):

        original_activity_name = flows.Activity.iloc[i]
        activity_prod = flows.Product.iloc[i]
        activity_database = flows.Database.iloc[i]

        if (
                original_activity_name,
                activity_prod,
                esm_location,
                esm_results_db_name
        ) in already_done:
            continue

        else:
            already_done.append((
                original_activity_name,
                activity_prod,
                esm_location,
                esm_results_db_name
            ))

        if (
                original_activity_name,
                activity_prod,
                esm_location,
                esm_results_db_name,
        ) in esm_results_db_dict_name:
            # if not, it means that the activity has not been created in the previous step
            # (e.g., no production, trivial results)

            new_activity = esm_results_db_dict_name[
                original_activity_name,
                activity_prod,
                esm_location,
                esm_results_db_name,
            ]

            for act in activities_of_esm_region:
                if act['name'] == original_activity_name and act['location'] == esm_location:
                    pass  # we do not want the new activity to be an input of itself
                else:
                    if create_new_db:
                        for exc in Dataset(act).get_technosphere_flows():
                            if (
                                    ((exc['name'] == original_activity_name)
                                     | (exc['name'] == original_activity_name.replace('market', 'market group')) 
                                     | (not update_exchanges_based_on_activity_name))
                                    & (exc['product'] == activity_prod)
                                    & (exc['database'] != esm_results_db_name)
                            ):
                                exc['name'] = new_activity['name']
                                exc['code'] = new_activity['code']
                                exc['database'] = esm_results_db_name
                                exc['input'] = (esm_results_db_name, new_activity['code'])
                                exc['location'] = esm_location
                    else:
                        act_bw = bd.Database(act['database']).get(act['code'])
                        k = 0  # exchange modification counter
                        for exc in [i for i in act_bw.technosphere()]:
                            if (
                                    ((exc['name'] == original_activity_name)
                                     | (exc['name'] == original_activity_name.replace('market', 'market group')) 
                                     | (not update_exchanges_based_on_activity_name))
                                    & (exc['product'] == activity_prod)
                                    & ((exc['database'] != esm_results_db_name)
                                       | (exc['input'][0] != esm_results_db_name))
                            ):
                                k += 1
                                exc['name'] = new_activity['name']
                                exc['code'] = new_activity['code']
                                exc['database'] = esm_results_db_name
                                exc['input'] = (esm_results_db_name, new_activity['code'])
                                exc['location'] = esm_location
                                exc.save()
                        if k > 0:  # if exchanges have been modified
                            act_bw.save()

            # Downstream activities of the original activity, if it exists for the ESM location
            for loc in locations:
                if (original_activity_name, activity_prod, loc, activity_database) in db_dict_name:
                    original_activity = db_dict_name[
                        original_activity_name, activity_prod, loc, activity_database
                    ]
                    downstream_consumers = Dataset(original_activity).get_downstream_consumers(db_as_list)
                    for act in downstream_consumers:
                        if act['name'] == original_activity_name and act['location'] == esm_location:
                            pass  # we do not want the new activity to be an input of itself
                        else:
                            if create_new_db:
                                for exc in Dataset(act).get_technosphere_flows():
                                    if (
                                            (exc['name'] == original_activity_name)
                                            & (exc['product'] == activity_prod)
                                            & (exc['location'] == loc)
                                            & (exc['database'] != esm_results_db_name)
                                    ):
                                        exc['code'] = new_activity['code']
                                        exc['database'] = esm_results_db_name
                                        exc['input'] = (esm_results_db_name, new_activity['code'])
                                        exc['location'] = esm_location

                            else:
                                act_bw = bd.Database(act['database']).get(act['code'])
                                k = 0  # exchange modification counter
                                for exc in [i for i in act_bw.technosphere()]:
                                    if (
                                            (exc['name'] == original_activity_name)
                                            & (exc['product'] == activity_prod)
                                            & (exc['location'] == loc)
                                            & ((exc['database'] != esm_results_db_name)
                                               | (exc['input'][0] != esm_results_db_name))
                                    ):
                                        k += 1
                                        exc['code'] = new_activity['code']
                                        exc['database'] = esm_results_db_name
                                        exc['input'] = (esm_results_db_name, new_activity['code'])
                                        exc['location'] = esm_location
                                        exc.save()

                                if k > 0:  # if exchanges have been modified
                                    act_bw.save()

    if specific_db_name is None:
        # Injecting local variables into the instance variables
        self.main_database.db_as_list = db_as_list

    if create_new_db:
        # Write the new database
        new_db = Database(db_as_list=db_as_list)
        new_db = new_db - esm_results_db
        new_db.write_to_brightway(new_db_name)


def _create_or_modify_activity_from_esm_results(
        self,
        original_activity_prod: str,
        original_activity_name: str,
        original_activity_database: str,
        flows: pd.DataFrame,
        esm_results: pd.DataFrame,
        new_end_use_types: pd.DataFrame,
) -> list[list[str]]:
    """
    Create or modify an activity in the LCI database based on the ESM results

    :param original_activity_prod: reference product of the original activity
    :param original_activity_name: name of the original activity
    :param original_activity_database: database of the original activity
    :param flows: mapping file between ESM flows and LCI datasets
    :param esm_results: results of the ESM in terms of annual production and installed capacity. It should contain the
        columns 'Name', 'Production', and 'Capacity'.
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :return: list of activities to perform double counting removal
    """

    # Store frequently accessed instance variables in local variables inside a method
    db_dict_name = self.main_database.db_as_dict_name
    db_dict_code = self.main_database.db_as_dict_code
    mapping = self.mapping
    esm_location = self.esm_location
    db_as_list = self.main_database.db_as_list
    unit_conversion = self.unit_conversion
    model = self.model
    esm_results_db_name = self.esm_results_db_name
    tech_to_remove_layers = self.tech_to_remove_layers

    # Check if the original activity is in the database for the location under study
    if (original_activity_name, original_activity_prod, esm_location, original_activity_database) in db_dict_name:
        original_activity = db_dict_name[
            original_activity_name, original_activity_prod, esm_location, original_activity_database
        ]

    # If not, we take a similar activity with another location and regionalize its foreground
    else:
        original_activity = [a for a in wurst.get_many(db_as_list, *[
            wurst.equals('name', original_activity_name),
            wurst.equals('reference product', original_activity_prod),
            wurst.equals('database', original_activity_database)
        ])][0]

        original_activity = self._regionalize_activity_foreground(act=original_activity)

    new_code = random_code()
    original_activity_unit = original_activity['unit']
    prod_flow = Dataset(original_activity).get_production_flow()
    prod_flow_amount = prod_flow['amount']  # can be -1 if it is a waste activity

    unit_conversion['LCA'] = unit_conversion['LCA'].apply(ecoinvent_unit_convention)
    unit_conversion['ESM'] = unit_conversion['ESM'].apply(ecoinvent_unit_convention)

    model['Flow'] = model.apply(lambda x: _replace_mobility_end_use_type(
        row=x,
        new_end_use_types=new_end_use_types
    ), axis=1)
    model = pd.merge(
        left=model,
        right=model[model.Amount == 1.0].drop(columns=['Amount']).rename(columns={'Flow': 'Output'}),
        how='left',
        on='Name'
    )

    act_to_flows_dict = {(flows['Product'].iloc[i], flows['Activity'].iloc[i]): list(
        flows[(flows['Product'] == flows['Product'].iloc[i])
              & (flows['Activity'] == flows['Activity'].iloc[i])]['Name']
    ) for i in range(len(flows))}

    flows_list = act_to_flows_dict[(original_activity_prod, original_activity_name)]
    end_use_tech_list = list(model[model.Output.isin(flows_list)].Name.unique())

    if len(end_use_tech_list) == 0:
        # Case where the layer has no production
        return []

    try:
        tech_to_remove_layers['Layers'] = tech_to_remove_layers['Layers'].apply(ast.literal_eval)
        tech_to_remove_layers['Technologies'] = tech_to_remove_layers['Technologies'].apply(ast.literal_eval)
    except ValueError:
        pass

    for i in range(len(tech_to_remove_layers)):
        if set(flows_list) == set(tech_to_remove_layers.Layers.iloc[i]):
            for tech in tech_to_remove_layers.Technologies.iloc[i]:
                end_use_tech_list.remove(tech)
        else:
            pass

    total_amount = 0  # initialize the total amount of production
    check_layers_mapping = False

    for tech in end_use_tech_list:

        if tech in list(esm_results.Name.unique()):
            amount = sum(esm_results[esm_results.Name == tech].Production)

        else:  # if the technology is not in the ESM results, we assume that its production is null
            amount = 0

        if tech in list(
                mapping[(mapping.Type == 'Operation') | (mapping.Type == 'Resource')].Name.unique()):
            total_amount += amount
            check_layers_mapping = True
        else:
            pass  # if the technology is not in the mapping file, we do not consider it in the result LCI dataset

    if check_layers_mapping is False:
        raise ValueError(f'The layer {flows_list} does not have any technology in the mapping file.')

    if total_amount == 0:  # no production in the layer
        return []

    exchanges = []
    perform_d_c = []

    for tech in end_use_tech_list:
        if tech in list(esm_results.Name.unique()):
            if self.operation_metrics_for_all_time_steps:
                amount_per_year = {}
                amount = 0
                for year in [y for y in self.list_of_years if y <= self.year]:
                    amount_per_year[year] = sum(esm_results[
                                                    (esm_results.Name == tech)
                                                    & (esm_results['Year_inst'] == year)
                                                ].Production)
                    amount += amount_per_year[year]
            else:
                amount = sum(esm_results[esm_results.Name == tech].Production)
        else:
            amount = 0
        if amount == 0:
            pass
        else:
            if not self.pathway:
                mapping.Year = None

            for year in [self.year] if not self.operation_metrics_for_all_time_steps \
                    else [y for y in self.list_of_years if y <= self.year]:

                if self.operation_metrics_for_all_time_steps:
                    amount = amount_per_year[year]

                if tech in list(
                        mapping[
                            ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                            & (True if not self.pathway else mapping.Year == year)
                        ].Name.unique()):
                    (activity_name, activity_prod, activity_database,
                     activity_location, activity_current_code, activity_new_code) = mapping[
                        (mapping.Name == tech)
                        & ((mapping.Type == 'Operation') | (mapping.Type == 'Resource'))
                        & (True if not self.pathway else mapping.Year == year)
                        ][['Activity', 'Product', 'Database', 'Location', 'Current_code', 'New_code']].values[0]

                    activity = db_dict_code[activity_database, activity_current_code]
                    activity_unit = activity['unit']

                    if activity_unit != original_activity_unit:
                        if original_activity_prod.split(',')[0] == 'transport':
                            conversion_factor = unit_conversion[
                                (unit_conversion.Name == tech)
                                & (unit_conversion.ESM == original_activity_unit)
                                & (unit_conversion.LCA == activity_unit)
                                ].Value.values
                        else:
                            conversion_factor = unit_conversion[
                                (unit_conversion.Name == original_activity_prod.split(',')[0])
                                & (unit_conversion.ESM == original_activity_unit)
                                & (unit_conversion.LCA == activity_unit)
                                ].Value.values
                        if len(list(set(conversion_factor))) == 0:
                            raise ValueError(f'The unit conversion factor between {activity_unit} and '
                                             f'{original_activity_unit} for {original_activity_prod.split(",")[0]} '
                                             f'is not in the unit conversion file.')
                        elif len(list(set(conversion_factor))) > 1:
                            raise ValueError(f'Multiple possible conversion factors between {activity_unit} and '
                                             f'{original_activity_unit} for {original_activity_prod.split(",")[0]}')
                        else:
                            amount *= conversion_factor[0]
                    else:
                        conversion_factor = 1.0

                    if prod_flow_amount == -1.0:  # for waste activities
                        amount *= -1.0

                    # Create new activity for the new exchange (because one activity may correspond to several ESM
                    # technologies, which might be adjusted later)
                    new_act = copy.deepcopy(activity)

                    if self.operation_metrics_for_all_time_steps:
                        new_act['name'] += f' ({tech}, {year})'
                        if year != self.year:
                            Dataset(new_act).relink(
                                name_database_unlink = [i['main_database'].db_names for i in self.time_steps
                                                        if i['year'] == year][0],
                                name_database_relink = [i['main_database'].db_names for i in self.time_steps
                                                        if i['year'] == self.year][0],
                                database_relink_as_list = db_as_list,
                                except_units = ['unit'],
                            )
                    else:
                        new_act['name'] += f' ({tech})'
                    new_act['code'] = activity_new_code
                    new_act['database'] = esm_results_db_name
                    prod_flow = Dataset(new_act).get_production_flow()
                    prod_flow['name'] = new_act['name']
                    prod_flow['code'] = activity_new_code
                    prod_flow['database'] = esm_results_db_name

                    if self.regionalize_foregrounds:
                        # Regionalize the foreground of the new activity
                        new_act = self._regionalize_activity_foreground(act=new_act)

                    db_as_list.append(new_act)
                    db_dict_name[(
                        new_act['name'],
                        new_act['reference product'],
                        new_act['location'],
                        new_act['database']
                    )] = new_act
                    db_dict_code[(new_act['database'], new_act['code'])] = new_act

                    new_exc = {
                        'amount': amount / total_amount,
                        'code': activity_new_code,
                        'type': 'technosphere',
                        'name': new_act['name'],
                        'product': activity_prod,
                        'unit': activity_unit,
                        'location': new_act['location'],
                        'database': esm_results_db_name,
                        'comment': f'{tech}, {conversion_factor}',
                    }
                    exchanges.append(new_exc)
                    if tech in list(mapping[mapping.Type == 'Operation'].Name.unique()):
                        # we only perform double counting removal for the operation activities
                        perform_d_c.append(
                            [tech, activity_prod, activity_name, activity_location, esm_results_db_name, activity_new_code]
                        )
                else:
                    self.logger.warning(f'The technology {tech} is not in the mapping file. '
                                        f'It cannot be considered in the result LCI dataset.')

    exchanges.append(
        {
            'amount': prod_flow_amount,
            'code': new_code,
            'type': 'production',
            'name': original_activity_name,
            'product': original_activity_prod,
            'unit': original_activity_unit,
            'location': esm_location,
            'database': esm_results_db_name,
        }
    )

    total_production_amount_original_activity = 0
    for exc in original_activity['exchanges']:
        if exc['unit'] not in [original_activity_unit, 'unit']:
            # Add flows to the new activity that are not production or construction flows
            exchanges.append(exc)
        if exc['type'] == 'technosphere' and exc['unit'] == original_activity_unit:
            total_production_amount_original_activity += exc['amount']

    losses_original_activity = total_production_amount_original_activity - 1
    if losses_original_activity > 0:  # add a loss coefficient
        exchanges.append(
            {
                'amount': losses_original_activity,
                'code': new_code,
                'type': 'technosphere',
                'name': original_activity_name,
                'product': original_activity_prod,
                'unit': original_activity_unit,
                'location': esm_location,
                'database': esm_results_db_name,
            }
        )

    new_activity = {
        'database': esm_results_db_name,
        'name': original_activity_name,
        'location': esm_location,
        'unit': original_activity_unit,
        'reference product': original_activity_prod,
        'code': new_code,
        'classifications': original_activity['classifications'],
        'comment': f'Activity derived from the ESM results in the layers {flows_list} for {esm_location}. '
                   + original_activity.get('comment', ''),
        'parameters': original_activity.get('parameters', {}),
        'categories': original_activity.get('categories', None),
        'exchanges': exchanges,
    }

    db_as_list.append(new_activity)

    # Injecting local variables into the instance variables
    self.main_database.db_as_list = db_as_list
    self.mapping = mapping

    return perform_d_c

def _correct_esm_and_lca_capacity_factor_differences(
        self,
        esm_results: pd.DataFrame,
        write_cp_report: bool = True,
) -> None:
    """
    Correct the differences of capacity factors between ESM technologies and their operation LCI datasets
    during the creation of the ESM results database. Concretely, it changes the amount of the construction input flow
    in the operation LCI dataset.

    :param esm_results: results of the ESM in terms of annual production and installed capacity. It should contain the
        columns 'Name', 'Production', and 'Capacity'.
    :param write_cp_report: if True, save a csv file reporting capacity factors differences in the results folder
    :return: None
    """

    db_dict_name = self.main_database.db_as_dict_name
    mapping = self.mapping
    esm_results_db_name = self.esm_results_db_name
    df_flows_set_to_zero = self.df_flows_set_to_zero
    unit_conversion = self.unit_conversion
    lifetime = self.lifetime
    tech_to_remove_layers = self.tech_to_remove_layers

    # readings lists as lists and not strings
    try:
        self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    technology_compositions_dict = {key: value for key, value in dict(zip(
        self.technology_compositions.Name, self.technology_compositions.Components
    )).items()}

    capacity_factor_report_list = []

    for tech in df_flows_set_to_zero.Name.unique():

        skip_tech = False

        if not self.pathway:
            mapping.Year = None
            esm_results.Year = None
        if not self.operation_metrics_for_all_time_steps:
            esm_results.Year_inst = None

        for year in [self.year] if not self.operation_metrics_for_all_time_steps \
                else [y for y in self.list_of_years if y <= self.year]:

            if skip_tech:
                continue

            if len(esm_results[
                       (esm_results.Name == tech)
                       & (True if not self.operation_metrics_for_all_time_steps else esm_results.Year_inst == year)
                   ]) == 0:
                # if the technology is not in the ESM results, we skip it
                continue

            elif tech in [tec for sublist in tech_to_remove_layers.Technologies for tec in sublist]:
                # if the technology is in the list of technologies to remove, we skip it
                continue

            act_to_adapt_list = []
            techno_flows_to_correct_dict = {}

            if tech not in technology_compositions_dict.keys():  # if the technology is not a composition
                # simple technologies are seen as compositions of one technology
                technology_compositions_dict[tech] = [tech]

            amount_constr_per_subcomp = {}

            for sub_comp in technology_compositions_dict[tech]:

                try:
                    unit_conversion_factor_constr = unit_conversion[
                        (unit_conversion.Name == sub_comp)
                        & (unit_conversion.Type == 'Construction')
                        ]['Value'].iloc[0]
                except IndexError:
                    self.logger.warning(f'No unit conversion factor for construction found for {sub_comp}. '
                                        f'The potential capacity factor difference cannot be corrected.')
                    skip_tech = True
                    continue

                if sub_comp != tech:
                    unit_conversion_factor_constr *= unit_conversion[
                        (unit_conversion.Name == tech)
                        & (unit_conversion.Type == 'Construction')
                    ]['Value'].iloc[0]

                lifetime_lca = lifetime[(lifetime.Name == sub_comp)]['LCA'].iloc[0]
                if pd.isna(lifetime_lca):
                    self.logger.warning(f'No LCA lifetime for {sub_comp}. Please provide a lifetime for this technology '
                                        f'in the lifetime csv file. Until then, the capacity factor difference cannot '
                                        f'be corrected.')
                    skip_tech = True
                    continue

                annual_production = esm_results[
                    (esm_results.Name == tech)
                    & (True if not self.operation_metrics_for_all_time_steps else esm_results.Year_inst == year)
                    & (True if not self.pathway else esm_results.Year == self.year)
                ]['Production'].iloc[0]

                installed_capacity = esm_results[
                    (esm_results.Name == tech)
                    & (True if not self.operation_metrics_for_all_time_steps else esm_results.Year_inst == year)
                    & (True if not self.pathway else esm_results.Year == self.year)
                ]['Capacity'].iloc[0]

                # amount_constr_esm is the amount of infrastructure unit to be used in the operation LCI dataset
                # given the annual production and installed capacity results of the ESM. This value can significantly differ
                # from the original value in the operation LCI dataset, due to differences in assumptions and operating modes.
                amount_constr_esm = installed_capacity * unit_conversion_factor_constr / (lifetime_lca * annual_production)
                amount_constr_per_subcomp[sub_comp] = amount_constr_esm

            if skip_tech:
                continue

            df_removed_construction_flows = df_flows_set_to_zero[
                (df_flows_set_to_zero.Name == tech)
                & (df_flows_set_to_zero.Unit == 'unit')
                & (True if not self.pathway else df_flows_set_to_zero.Year == year)
            ]

            for idx, row in df_removed_construction_flows.iterrows():

                if row['Activity'] == f'{tech}, Operation':
                    act_name = mapping[(mapping.Name == tech) & (mapping.Type == 'Operation')]['Activity'].iloc[0]

                    if self.operation_metrics_for_all_time_steps:
                        act_name += f' ({tech}, {year})'
                    else:
                        act_name += f' ({tech})'

                    if (
                            act_name,
                            row['Product'],
                            row['Location'],
                            esm_results_db_name,
                    ) in db_dict_name:
                        act_to_adapt = db_dict_name[(
                            act_name,
                            row['Product'],
                            row['Location'],
                            esm_results_db_name,
                        )]
                    else:  # i.e., the technology is not used in the ESM configuration
                        act_to_adapt = None

                else:
                    if (
                            row['Activity'],
                            row['Product'],
                            row['Location'],
                            esm_results_db_name,
                    ) in db_dict_name:
                        act_to_adapt = db_dict_name[(
                            row['Activity'],
                            row['Product'],
                            row['Location'],
                            esm_results_db_name,
                        )]
                    else:  # i.e., the technology is not used in the ESM configuration
                        act_to_adapt = None

                if act_to_adapt is not None and act_to_adapt not in act_to_adapt_list:  # avoid to apply correction several times
                    act_to_adapt_list.append(act_to_adapt)
                    techno_flows_to_correct_dict[
                        (act_to_adapt['database'], act_to_adapt['code'])
                    ] = []

                if act_to_adapt is not None:
                    act_exc = db_dict_name[(
                        row['Removed flow activity'],
                        row['Removed flow product'],
                        row['Removed flow location'],
                        row['Removed flow database'],
                    )]
                    techno_flows_to_correct_dict[
                        (act_to_adapt['database'], act_to_adapt['code'])
                    ] += [(act_exc['database'], act_exc['code'])]

            for act in act_to_adapt_list:

                for exc in Dataset(act).get_technosphere_flows():
                    i = 0
                    if (exc['database'], exc['code']) in techno_flows_to_correct_dict[(act['database'], act['code'])]:
                        for sub_comp in technology_compositions_dict[tech]:
                            if exc['product'] == mapping[(mapping.Name == sub_comp)
                                                         & (mapping.Type == 'Construction')].Product.iloc[0]:
                                i+=1
                                amount_constr_esm = amount_constr_per_subcomp[sub_comp]
                                amount_constr_lca = exc['amount']  # original infrastructure amount in the operation LCI dataset
                                exc['amount'] = amount_constr_esm  # we replace the latter by the one derived from ESM results
                                exc['comment'] = (f'TF multiplied by {round(amount_constr_esm / amount_constr_lca, 4)} (capacity '
                                                  f'factor). ' + exc.get('comment', ''))
                                if amount_constr_lca == 0:
                                    print(act['name'], exc['name'])

                                capacity_factor_report_list.append([
                                    tech,
                                    exc['name'],
                                    exc['product'],
                                    exc['location'],
                                    exc['database'],
                                    exc['code'],
                                    amount_constr_lca,
                                    amount_constr_esm,
                                ])  # reporting capacity factors differences
                    if i > 1:
                        self.logger.warning(f"Exchange {exc['name']} in activity {act['name']} has matched with several "
                                         f"sub-components of technology {tech}.")
                act['comment'] = (f'Infrastructure flows have been harmonized with the ESM to account for capacity factor '
                                  f'differences. ') + act.get('comment', '')

    if write_cp_report:
        pd.DataFrame(
            data=capacity_factor_report_list,
            columns=['Name', 'Product', 'Activity', 'Location', 'Database', 'Code', 'Amount LCA', 'Amount ESM'],
        ).to_csv(f"{self.results_path_file}capacity_factor_differences.csv", index=False)

@staticmethod
def _replace_mobility_end_use_type(row: pd.Series, new_end_use_types: pd.DataFrame) -> str:
    """
    Reformat the end use type of the mobility technologies

    :param row: row of the model dataframe
    :param new_end_use_types: adapt end use types to fit the results LCI datasets mapping
    :return: updated end use type
    """

    for i in range(len(new_end_use_types)):
        name = new_end_use_types.Name.iloc[i]
        old_eut = new_end_use_types.Old.iloc[i]
        new_eut = new_end_use_types.New.iloc[i]
        search_type = new_end_use_types['Search type'].iloc[i]
        if search_type == 'startswith':
            if (row['Name'].startswith(name)) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        elif search_type == 'contains':
            if (name in row['Name']) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        elif search_type == 'equals':
            if (name == row['Name']) & (old_eut in row['Flow']) & (row['Amount'] == 1.0):
                return new_eut
        else:
            raise ValueError('The search type should be either "startswith", "contains" or "equals".')

    return row['Flow']
