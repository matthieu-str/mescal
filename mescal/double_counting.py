import pandas as pd
import ast
import copy
import wurst
from .modify_inventory import change_dac_biogenic_carbon_flow, change_fossil_carbon_flows_of_biofuels
from .adapt_efficiency import correct_esm_and_lca_efficiency_differences
from .database import Database, Dataset
from .utils import random_code


class ESMDatabase:
    """
    Create the ESM database after double counting removal. Three csv files summarizing the double-counting removal
        process are automatically saved in the results' folder: double_counting_removal.csv (amount of removed flows),
        double_counting_removal_count.csv (number of flows set to zero), and removed_flows_list.csv (specific activities
        in which the flows were removed).
    """

    def __init__(
            self,
            mapping: pd.DataFrame,
            model: pd.DataFrame,
            mapping_esm_flows_to_CPC_cat: pd.DataFrame,
            main_database: Database,
            esm_db_name: str,
            technology_compositions: pd.DataFrame = None,
            results_path_file: str = 'results/',
            tech_specifics: pd.DataFrame = None,
            regionalize_foregrounds: bool = False,
            accepted_locations: list[str] = None,
            target_region: str = None,
            locations_ranking: list[str] = None,
            spatialized_database: bool = False,
            spatialized_biosphere_db: Database = None,
            write_database: bool = True,
            return_obj: str = 'mapping',
            efficiency: pd.DataFrame = None,
            unit_conversion: pd.DataFrame = None,
    ):
        """
        Initialize the ESM database creation

        :param mapping: mapping file
        :param model: model file
        :param tech_specifics: technology specifics
        :param technology_compositions: technology compositions
        :param mapping_esm_flows_to_CPC_cat: mapping file between the ESM flows and the CPC categories
        :param main_database: LCI database
        :param esm_db_name: name of the new LCI database
        :param results_path_file: path to the results folder
        :param regionalize_foregrounds: if True, regionalize the foreground activities
        :param accepted_locations: list of regions to keep in case of regionalization
        :param target_region: target region in case of regionalization
        :param locations_ranking: ranking of the preferred locations in case of regionalization
        :param spatialized_database: if True, the main database has spatialized elementary flows
        :param spatialized_biosphere_db: list of flows in the spatialized biosphere database
        :param write_database: if True, write the esm database to the brightway project and saves double-counting report
            files
        :param return_obj: if 'mapping', return the mapping file as a pd.DataFrame, if 'database', return the ESM database
            as a list of dictionaries
        :param efficiency: file containing the ESM (Name, Flow) couples to correct regarding efficiency differences between
            the ESM and LCI database
        :param unit_conversion: file containing unit conversion factors
        :return: mapping file (updated with new codes) or the ESM database as a list of dictionaries, depending on the
            'return_obj' parameter. Three csv files are also automatically saved in the results' folder.
        """
        self.mapping = mapping
        self.model = model
        self.tech_specifics = tech_specifics
        self.technology_compositions = technology_compositions
        self.mapping_esm_flows_to_CPC_cat = mapping_esm_flows_to_CPC_cat
        self.main_database = main_database
        self.esm_db_name = esm_db_name
        self.results_path_file = results_path_file
        self.regionalize_foregrounds = regionalize_foregrounds
        self.accepted_locations = accepted_locations
        self.target_region = target_region
        self.locations_ranking = locations_ranking
        self.spatialized_database = spatialized_database
        self.spatialized_biosphere_db = spatialized_biosphere_db
        self.write_database = write_database
        self.return_obj = return_obj
        self.efficiency = efficiency
        self.unit_conversion = unit_conversion

    @property
    def mapping_op(self):
        mapping_op = self.mapping[self.mapping['Type'] == 'Operation']
        model_pivot = self.model.pivot(index='Name', columns='Flow', values='Amount').reset_index()
        model_pivot.fillna(0, inplace=True)
        mapping_op = pd.merge(mapping_op, model_pivot, on='Name', how='left')
        mapping_op['CONSTRUCTION'] = mapping_op.shape[0] * [0]
        mapping_op = add_technology_specifics(mapping_op, self.tech_specifics)
        return mapping_op

    @property
    def mapping_constr(self):
        return self.mapping[self.mapping['Type'] == 'Construction']

    @property
    def mapping_res(self):
        return self.mapping[self.mapping['Type'] == 'Resource']

    @property
    def activities_background_search(self):
        return list(self.tech_specifics[self.tech_specifics.Specifics == 'Background search'].Name)

    @property
    def background_search_act(self):
        background_search_act = {}
        for tech in self.activities_background_search:
            background_search_act[tech] = self.tech_specifics[self.tech_specifics.Name == tech].Amount.iloc[0]
        return background_search_act

    @property
    def no_construction_list(self):
        return list(self.tech_specifics[self.tech_specifics.Specifics == 'No construction'].Name)

    @property
    def no_background_search_list(self):
        return list(self.tech_specifics[self.tech_specifics.Specifics == 'No background search'].Name)

    @property
    def import_export_list(self):
        return list(self.tech_specifics[self.tech_specifics.Specifics == 'Import/Export'].Name)

    @property
    def technology_compositions_dict(self):
        return {key: value for key, value in dict(zip(
            self.technology_compositions.Name, self.technology_compositions.Components
        )).items()}

    from .regionalization import regionalize_activity_foreground, change_location_activity, change_location_mapping_file

    def create_esm_database(self) -> pd.DataFrame | list[dict]:

        if self.tech_specifics is None:
            self.tech_specifics = pd.DataFrame(columns=['Name', 'Specifics', 'Amount'])
        if self.technology_compositions is None:
            self.technology_compositions = pd.DataFrame(columns=['Name', 'Components'])

        try:
            self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
        except ValueError:
            pass

        if (self.efficiency is not None) & (self.unit_conversion is None):
            raise ValueError('Unit conversion file is needed for efficiency differences correction. Please provide it.')

        # Adding current code to the mapping file
        self.mapping['Current_code'] = self.mapping.apply(lambda row: self.main_database.get_code(
            product=row['Product'],
            activity=row['Activity'],
            location=row['Location'],
            database=row['Database']
        ), axis=1)

        # Creating a new code for each activity to be added
        self.mapping['New_code'] = self.mapping.apply(lambda row: random_code(), axis=1)

        N = self.mapping.shape[1]

        # Add construction and resource activities to the database (which do not need double counting removal)
        self.main_database = self.add_activities_to_database(act_type='Construction')
        self.main_database = self.add_activities_to_database(act_type='Resource')

        flows_set_to_zero, ei_removal = self.double_counting_removal(N=N, ESM_inputs='all')

        if self.write_database:

            df_flows_set_to_zero = pd.DataFrame(data=flows_set_to_zero,
                                                columns=[
                                                    'Name', 'Product', 'Activity', 'Location', 'Database', 'Code',
                                                    'Amount',
                                                    'Unit', 'Removed flow product', 'Removed flow activity',
                                                    'Removed flow location', 'Removed flow database',
                                                    'Removed flow code'
                                                ])
            df_flows_set_to_zero.drop_duplicates(inplace=True)

            ei_removal_amount = {}
            ei_removal_count = {}
            for tech in list(self.mapping_op.Name):
                ei_removal_amount[tech] = {}
                ei_removal_count[tech] = {}
                for res in list(self.mapping_op.iloc[:, N:].columns):
                    ei_removal_amount[tech][res] = ei_removal[tech][res]['amount']
                    ei_removal_count[tech][res] = ei_removal[tech][res]['count']

            double_counting_removal_amount = pd.DataFrame.from_dict(ei_removal_amount, orient='index')
            double_counting_removal_count = pd.DataFrame.from_dict(ei_removal_count, orient='index')

            double_counting_removal_amount.reset_index(inplace=True)
            double_counting_removal_amount = double_counting_removal_amount.melt(
                id_vars=['index'],
                value_vars=double_counting_removal_amount.columns[1:]
            )
            double_counting_removal_amount.rename(columns={'index': 'Name', 'variable': 'Flow', 'value': 'Amount'},
                                                  inplace=True)
            double_counting_removal_amount.drop(
                double_counting_removal_amount[double_counting_removal_amount.Amount == 0].index, inplace=True
            )

            double_counting_removal_count.reset_index(inplace=True)
            double_counting_removal_count = double_counting_removal_count.melt(
                id_vars=['index'],
                value_vars=double_counting_removal_count.columns[1:]
            )
            double_counting_removal_count.rename(columns={'index': 'Name', 'variable': 'Flow', 'value': 'Amount'},
                                                 inplace=True)
            double_counting_removal_count.drop(
                double_counting_removal_count[double_counting_removal_count.Amount == 0].index, inplace=True
            )

            if self.efficiency is not None:
                main_database = correct_esm_and_lca_efficiency_differences(
                    db=main_database,
                    model=model,
                    efficiency=efficiency,
                    mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC_cat,
                    removed_flows=df_flows_set_to_zero,
                    unit_conversion=unit_conversion,
                    double_counting_removal=double_counting_removal_amount,
                )

            double_counting_removal_amount.to_csv(f"{self.results_path_file}double_counting_removal.csv", index=False)
            double_counting_removal_count.to_csv(f"{self.results_path_file}double_counting_removal_count.csv", index=False)
            df_flows_set_to_zero.to_csv(f"{self.results_path_file}removed_flows_list.csv", index=False)

            esm_db = Database(db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])
            esm_db.write_to_brightway(self.esm_db_name)

            # Change carbon flow of DAC from biogenic to fossil
            dac_technologies = list(self.tech_specifics[self.tech_specifics.Specifics == 'DAC'].Name)
            for tech in dac_technologies:
                if f'{tech}, Operation' in [act['name'] for act in esm_db.db_as_list]:
                    change_dac_biogenic_carbon_flow(db_name=self.esm_db_name, activity_name=f'{tech}, Operation')

            # Change carbon flows of biofuel mobility technologies
            biofuel_mob_tech = self.tech_specifics[self.tech_specifics.Specifics == 'Biofuel'][['Name', 'Amount']].values.tolist()
            for tech, biogenic_ratio in biofuel_mob_tech:
                if f'{tech}, Operation' in [act['name'] for act in esm_db.db_as_list]:
                    change_fossil_carbon_flows_of_biofuels(
                        db_name=self.esm_db_name,
                        activity_name=f'{tech}, Operation',
                        biogenic_ratio=biogenic_ratio
                    )

        if self.return_obj == 'mapping':
            return self.mapping
        elif self.return_obj == 'database':
            return [act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name]
        else:
            raise ValueError("return_obj must be 'mapping' or 'database'")

    def add_activities_to_database(
            self,
            act_type: str,
    ) -> Database:
        """
        Add new activities to the LCI database

        :param act_type: can be 'Construction', 'Operation', or 'Resource'
        :return: updated LCI database
        """
        mapping_type = self.mapping[self.mapping['Type'] == act_type]
        for i in range(len(mapping_type)):
            ds = self.create_new_activity(
                name=mapping_type['Name'].iloc[i],
                act_type=act_type,
                current_code=mapping_type['Current_code'].iloc[i],
                new_code=mapping_type['New_code'].iloc[i],
                database_name=mapping_type['Database'].iloc[i],
            )
            self.main_database.db_as_list.append(ds)
        return self.main_database

    def create_new_activity(
            self,
            name: str,
            act_type: str,
            current_code: str,
            new_code: str,
            database_name: str,
    ) -> dict:
        """
        Create a new LCI dataset for the ESM technology or resource

        :param name: name of the technology or resource in the esm
        :param act_type: can be 'Construction', 'Operation', or 'Resource'
        :param current_code: code of the activity in the original LCI database
        :param new_code: code of the new activity in the new LCI database
        :param database_name: name of the original LCI database
        :return: new LCI dataset for the technology or resource
        """

        act = self.main_database.db_as_dict_code[(database_name, current_code)]

        new_act = copy.deepcopy(act)
        new_act['name'] = f'{name}, {act_type}'
        new_act['code'] = new_code
        new_act['database'] = self.esm_db_name
        prod_flow = Dataset(new_act).get_production_flow()
        prod_flow['name'] = f'{name}, {act_type}'
        prod_flow['code'] = new_code
        prod_flow['database'] = self.esm_db_name

        if self.regionalize_foregrounds:
            new_act = self.regionalize_activity_foreground(act=new_act)

        return new_act

    def background_search(
            self,
            act: dict,
            k: int,
            k_lim: int,
            amount: float,
            explore_type: str,
            ESM_inputs: list[str] or str,
            perform_d_c: list[list],
            create_new_db: bool = True
    ) -> list[list]:
        """
        Explores the tree of the market activity with a recursive approach and write the activities to actually check for
        double-counting in the list perform_d_c.

        :param act: activity
        :param k: tree depth of act with respect to starting activity
        :param k_lim: maximum allowed tree depth (i.e., maximum recursion depth)
        :param amount: product of amounts when going down in the tree
        :param explore_type: can be 'market' or 'background_removal'
        :param ESM_inputs: list of the ESM flows to perform double counting removal on
        :param perform_d_c: list of activities to check for double counting
        :param create_new_db: if True, create a new database
        :return: list of activities to check for double counting
        """

        if explore_type == 'market':
            # we want to test whether the activity is a market (if yes, we explore the background),
            # thus the condition is False a priori
            condition = False
        elif explore_type == 'background_removal':
            # we know that we want to explore the background, thus we directly set the condition as true
            condition = True
        else:
            raise ValueError('Type should be either "market" or "background_removal"')

        if 'CPC' in dict(act['classifications']).keys():  # we have a CPC category
            CPC_cat = dict(act['classifications'])['CPC']
        else:
            raise ValueError(f'No CPC category in the activity ({act["database"]}, {act["code"]})')

        db_dict_code = self.main_database.db_as_dict_code
        technosphere_flows = Dataset(act).get_technosphere_flows()  # technosphere flows of the activity
        technosphere_flows_act = [db_dict_code[(flow['database'], flow['code'])] for flow in
                                  technosphere_flows]  # activities corresponding to technosphere flows inputs
        biosphere_flows = Dataset(act).get_biosphere_flows()  # biosphere flows of the activity

        # Here, we try different conditions to assess whether we have to explore the background of the activity or not
        if not condition:
            if len(technosphere_flows) == 1:  # not truly a market but we have to go to the next layer
                techno_act = technosphere_flows_act[0]
                if (techno_act['unit'] == act['unit']) & (technosphere_flows[0]['amount'] == 1):  # same unit and quantity
                    condition = True
                    if 'classifications' in techno_act.keys():  # adding a CPC cat if not already the case
                        if 'CPC' in dict(techno_act['classifications']).keys():
                            pass
                        else:
                            techno_act['classifications'].append(('CPC', CPC_cat))
                    else:
                        techno_act['classifications'] = [('CPC', CPC_cat)]

        if not condition:
            # if all the technosphere flows have the same name
            if (len(technosphere_flows) > 1) & (len(list(set([flow['name'] for flow in technosphere_flows]))) == 1):
                condition = True
                if sum(['classifications' in j.keys() for j in technosphere_flows_act]) == len(
                        technosphere_flows_act):  # adding a CPC cat if not already the case
                    if len(technosphere_flows_act) == sum(
                            ['CPC' in [j['classifications'][i][0]] for j in technosphere_flows_act for i in
                             range(len(j['classifications']))]):
                        pass
                    else:
                        for j in technosphere_flows_act:
                            if 'CPC' in dict(j['classifications']).keys():
                                pass
                            else:
                                j['classifications'].append(('CPC', CPC_cat))
                else:
                    for j in technosphere_flows_act:
                        if 'classifications' in j.keys():
                            if 'CPC' in dict(j['classifications']).keys():
                                pass
                            else:
                                j['classifications'].append(('CPC', CPC_cat))
                        else:
                            j['classifications'] = [('CPC', CPC_cat)]

        if not condition:
            if 'market for' in act['name']:
                condition = True

        if not condition:
            if 'activity type' in act.keys():
                if 'market activity' == act['activity type']:  # is the activity type is "market"
                    condition = True

        if not condition:
            if (len(technosphere_flows) > 1) & (sum(['classifications' in j.keys() for j in technosphere_flows_act])
                                                == len(technosphere_flows_act)):  # all technosphere flows have the classification key
                # if all technosphere flows have the CPC category in their classifications
                if len(technosphere_flows_act) == sum(['CPC' in [j['classifications'][i][0]]
                                                       for j in technosphere_flows_act
                                                       for i in range(len(j['classifications']))]):
                    # CPC categories corresponding to technosphere flows:
                    technosphere_flows_CPC = [dict(j['classifications'])['CPC'] for j in technosphere_flows_act]
                    # if there is one CPC category among all technosphere flows and no direct emissions
                    if (len(set(technosphere_flows_CPC)) == 1) & (len(biosphere_flows) == 0):
                        condition = True

        # The tests to assess whether the activity background should be explored stops here.
        # If one condition was fulfilled, we continue to explore the tree.
        if condition:
            for flow in technosphere_flows:
                if (create_new_db is True) & (flow['database'] == self.esm_db_name):
                    # this means that the same activity is several times in the tree, thus inducing an infinite loop
                    pass
                else:
                    if explore_type == 'market':
                        techno_act = db_dict_code[(flow['database'], flow['code'])]
                        if 'classifications' in techno_act.keys():
                            if 'CPC' in dict(techno_act['classifications']):
                                CPC_cat_new = dict(techno_act['classifications'])['CPC']
                                if CPC_cat == CPC_cat_new:
                                    if create_new_db:
                                        # Modify and save the activity in the ESM database
                                        new_act = copy.deepcopy(techno_act)
                                        new_code = random_code()
                                        new_act['database'] = self.esm_db_name
                                        new_act['code'] = new_code
                                        prod_flow = Dataset(new_act).get_production_flow()
                                        prod_flow['code'] = new_code
                                        prod_flow['database'] = self.esm_db_name
                                        self.main_database.db_as_list.append(new_act)
                                        # db_dict_name[(new_act['name'], new_act['reference product'], new_act['location'],
                                        #               new_act['database'])] = new_act
                                        # db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                        # Modify the flow between the activity and its inventory
                                        flow['database'] = self.esm_db_name
                                        flow['code'] = new_code
                                        flow['input'] = (self.esm_db_name, new_code)
                                    else:
                                        new_act = techno_act

                                    if k < k_lim:  # we continue until maximum depth is reached:
                                        perform_d_c, db, db_dict_code, db_dict_name = self.background_search(
                                            act=new_act,
                                            k=k + 1,
                                            k_lim=k_lim,
                                            amount=amount * flow['amount'],
                                            explore_type='market',
                                            ESM_inputs=ESM_inputs,
                                            perform_d_c=perform_d_c,
                                            create_new_db=create_new_db,
                                        )
                                        # adding 1 to the current depth k and multiply amount by the flow's amount
                                    else:
                                        # if the limit is reached, we consider the last activity for double counting removal
                                        new_act['comment'] = (f"Subject to double-counting removal ({explore_type}). "
                                                              + new_act.get('comment', ''))
                                        perform_d_c.append(
                                            [new_act['name'], new_act['code'], amount * flow['amount'], k + 1, ESM_inputs]
                                        )
                        else:
                            pass
                    elif explore_type == 'background_removal':
                        if (
                                (flow['amount'] > 0)
                                & (flow['unit'] not in ['unit', 'megajoule', 'kilowatt hour', 'ton kilometer'])
                                & (flow['product'] not in ['tap water'])
                        ):
                            # we do not consider construction, transport and energy flows (we typically target fuel flows
                            # in kg or m3) as well as negative flows
                            techno_act = db_dict_code[(flow['database'], flow['code'])]
                            if 'classifications' in techno_act.keys():
                                if 'CPC' in dict(techno_act['classifications']):
                                    if create_new_db:
                                        new_act = copy.deepcopy(techno_act)
                                        new_code = random_code()
                                        new_act['database'] = self.esm_db_name
                                        new_act['code'] = new_code
                                        prod_flow = Dataset(new_act).get_production_flow()
                                        prod_flow['code'] = new_code
                                        prod_flow['database'] = self.esm_db_name
                                        self.main_database.db_as_list.append(new_act)
                                        # db_dict_name[(new_act['name'], new_act['reference product'], new_act['location'],
                                        #               new_act['database'])] = new_act
                                        # db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                        # Modify the flow between the activity and its inventory and save it
                                        flow['database'] = self.esm_db_name
                                        flow['code'] = new_code
                                        flow['input'] = (self.esm_db_name, new_code)
                                    else:
                                        new_act = techno_act

                                    if k < k_lim:  # we continue until maximum depth is reached:
                                        perform_d_c, db, db_dict_code, db_dict_name = self.background_search(
                                            act=new_act,
                                            k=k+1,
                                            k_lim=k_lim,
                                            amount=amount*flow['amount'],
                                            explore_type='market',
                                            ESM_inputs=ESM_inputs,
                                            perform_d_c=perform_d_c,
                                            create_new_db=create_new_db,
                                        )
                                        # here we want to check whether the next activity is a market or not, if not,
                                        # the activity will be added for double counting
                                    else:
                                        # if the limit is reached, we consider the last activity for double counting removal
                                        new_act['comment'] = (f"Subject to double-counting removal ({explore_type}). "
                                                              + new_act.get('comment', ''))
                                        perform_d_c.append([new_act['name'], new_act['code'],
                                                            amount*flow['amount'], k+1, ESM_inputs])
                                else:
                                    raise ValueError(f"No CPC cat: ({techno_act['database']}, {techno_act['code']})")
                            else:
                                raise ValueError(f"No CPC cat: ({techno_act['database']}, {techno_act['code']})")
            return perform_d_c

        else:  # the activity is not a market, thus it is added to the list for double-counting removal
            act['comment'] = f"Subject to double-counting removal ({explore_type}). " + act.get('comment', '')
            perform_d_c.append([act['name'], act['code'], amount, k, ESM_inputs])
            return perform_d_c

    def double_counting_removal(
            self,
            N: int,
            ESM_inputs: list[str] or str = 'all',
            create_new_db: bool = True,
    ) -> tuple[list[list], dict]:
        """
        Remove double counting in the ESM database and write it in the brightway project

        :param N: number of columns of the original mapping file
        :param ESM_inputs: list of the ESM flows to perform double counting removal on
        :param create_new_db: if True, create a new database
        :return: list of removed flows, dictionary of removed quantities
        """
        # Initializing list of removed flows
        flows_set_to_zero = []

        # Initializing the dict of removed quantities
        ei_removal = {}
        for tech in list(self.mapping_op.Name):
            ei_removal[tech] = {}
            for res in list(self.mapping_op.iloc[:, N:].columns):
                ei_removal[tech][res] = {}
                ei_removal[tech][res]['amount'] = 0
                ei_removal[tech][res]['count'] = 0

        # readings lists as lists and not strings
        try:
            self.mapping_esm_flows_to_CPC_cat.CPC = self.mapping_esm_flows_to_CPC_cat.CPC.apply(ast.literal_eval)
        except ValueError:
            pass

        # inverse mapping dictionary (i.e., from CPC categories to the ESM flows)
        mapping_esm_flows_to_CPC_dict = {key: value for key, value in dict(zip(
            self.mapping_esm_flows_to_CPC_cat.Flow, self.mapping_esm_flows_to_CPC_cat.CPC
        )).items()}
        mapping_CPC_to_esm_flows_dict = {}
        for k, v in mapping_esm_flows_to_CPC_dict.items():
            for x in v:
                mapping_CPC_to_esm_flows_dict.setdefault(x, []).append(k)

        for i in range(len(self.mapping_op)):
            tech = self.mapping_op['Name'].iloc[i]  # name of ES technology
            # print(tech)

            # Initialization of the list of construction activities and corresponding CPC categories
            act_constr_list = []
            CPC_constr_list = []
            mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] = []

            # Construction activity
            if tech in self.no_construction_list:
                pass

            else:
                if tech not in self.technology_compositions_dict.keys():  # if the technology is a composition
                    # simple technologies are seen as compositions of one technology
                    self.technology_compositions_dict[tech] = [tech]

                for sub_comp in self.technology_compositions_dict[tech]:  # looping over the subcomponents of the composition

                    database_constr = self.mapping_constr[self.mapping_constr.Name == sub_comp]['Database'].iloc[0]
                    current_code_constr = self.mapping_constr[self.mapping_constr.Name == sub_comp]['Current_code'].iloc[0]

                    act_constr = self.main_database.db_as_dict_code[(database_constr, current_code_constr)]
                    act_constr_list.append(act_constr)
                    CPC_constr = dict(act_constr['classifications'])['CPC']
                    CPC_constr_list.append(CPC_constr)
                    mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] += [CPC_constr]
                    mapping_CPC_to_esm_flows_dict[CPC_constr] = ['OWN_CONSTRUCTION']

            # Operation activity
            database_op = self.mapping_op['Database'].iloc[i]  # LCA database of the operation technology
            current_code_op = self.mapping_op['Current_code'].iloc[i]  # code in ecoinvent

            # identification of the activity in ecoinvent database
            act_op = self.main_database.db_as_dict_code[(database_op, current_code_op)]

            if create_new_db:
                # Copy the activity and change the database (no new activity in original ecoinvent database)
                new_code = self.mapping_op['New_code'].iloc[i]  # new code defined previously
                new_act_op = copy.deepcopy(act_op)
                new_act_op['code'] = new_code
                new_act_op['database'] = self.esm_db_name
                prod_flow = Dataset(new_act_op).get_production_flow()
                prod_flow['code'] = new_code
                prod_flow['database'] = self.esm_db_name
                self.main_database.db_as_list.append(new_act_op)
                # self.main_database.db_as_dict_name[
                #     (new_act_op['name'], new_act_op['reference product'],
                #      new_act_op['location'], new_act_op['database'])
                # ] = new_act_op
                # self.main_database.db_as_dict_code[(new_act_op['database'], new_act_op['code'])] = new_act_op
            else:
                new_act_op = act_op

            if tech in self.no_background_search_list:
                new_act_op['comment'] = f"Subject to double-counting removal. " + new_act_op.get('comment', '')
                perform_d_c = [[new_act_op['name'], new_act_op['code'], 1, 0, ESM_inputs]]
            else:
                perform_d_c = self.background_search(
                    act=new_act_op,
                    k=0,
                    k_lim=10,
                    amount=1,
                    explore_type='market',
                    ESM_inputs=ESM_inputs,
                    perform_d_c=[],
                    create_new_db=create_new_db,
                )  # list of activities to perform double counting removal on

            if create_new_db:
                new_act_op['name'] = f'{tech}, Operation'  # saving name after market identification
                prod_flow = Dataset(new_act_op).get_production_flow()
                prod_flow['name'] = f'{tech}, Operation'

            id_d_c = 0
            while id_d_c < len(perform_d_c):

                new_act_op_d_c_code = perform_d_c[id_d_c][1]  # activity code
                new_act_op_d_c_amount = perform_d_c[id_d_c][2]  # multiplying factor as we went down in the tree
                k_deep = perform_d_c[id_d_c][3]  # depth level in the process tree
                new_act_op_d_c = None

                if (self.esm_db_name, new_act_op_d_c_code) in self.main_database.db_as_dict_code:
                    new_act_op_d_c = self.main_database.db_as_dict_code[(self.esm_db_name, new_act_op_d_c_code)]  # activity in the database
                else:
                    db_names_list = list(set([a['database'] for a in self.main_database.db_as_list]))
                    for db_name in db_names_list:
                        if (db_name, new_act_op_d_c_code) in self.main_database.db_as_dict_code:
                            new_act_op_d_c = self.main_database.db_as_dict_code[(db_name, new_act_op_d_c_code)]

                if new_act_op_d_c is None:
                    raise ValueError(f"Activity not found: {new_act_op_d_c_code}")

                if self.regionalize_foregrounds:
                    self.main_database.db_as_list.remove(new_act_op_d_c)
                    new_act_op_d_c = self.regionalize_activity_foreground(
                        act=new_act_op_d_c,
                    )
                    self.main_database.db_as_list.append(new_act_op_d_c)
                    self.main_database.db_as_dict_name[
                        (new_act_op_d_c['name'], new_act_op_d_c['reference product'],
                         new_act_op_d_c['location'], new_act_op_d_c['database'])
                    ] = new_act_op_d_c

                if perform_d_c[id_d_c][4] == 'all':
                    # list of inputs in the ESM (i.e., negative flows in layers_in_out)
                    ES_inputs = list(self.mapping_op.iloc[:, N:].iloc[i][self.mapping_op.iloc[:, N:].iloc[i] < 0].index)
                else:
                    ES_inputs = perform_d_c[id_d_c][4]

                # CPCs corresponding to the ESM list of inputs
                CPC_inputs = list(mapping_esm_flows_to_CPC_dict[inp] for inp in ES_inputs)
                CPC_inputs = [item for sublist in CPC_inputs for item in sublist]  # flatten the list of lists

                # Creating the list containing the CPCs of all technosphere flows of the activity
                technosphere_inputs = Dataset(new_act_op_d_c).get_technosphere_flows()
                technosphere_inputs_CPC = []

                for exc in technosphere_inputs:

                    # if (exc['amount'] < 0) & (exc['unit'] == 'unit'):
                    #     print('Potential decommission flow:', new_act_op_d_c['name'], exc['name'], exc['amount'])

                    database = exc['database']
                    code = exc['code']
                    act_flow = self.main_database.db_as_dict_code[(database, code)]

                    if 'classifications' in list(act_flow.keys()):
                        if 'CPC' in dict(act_flow['classifications']).keys():
                            technosphere_inputs_CPC.append(dict(act_flow['classifications'])['CPC'])
                        else:
                            technosphere_inputs_CPC.append('None')
                    else:
                        technosphere_inputs_CPC.append('None')

                # Finding the indices of technosphere flows that are also in the ESM inputs
                # (i.e., flows that we want to put to zero)
                set_CPC_inputs = set(CPC_inputs)
                id_technosphere_inputs_zero = [i for i, e in enumerate(technosphere_inputs_CPC) if e in set_CPC_inputs]

                if tech in self.no_construction_list:
                    pass
                elif perform_d_c[id_d_c][4] != 'all':
                    pass
                else:
                    comp_condition = True
                    for n in range(len(CPC_constr_list)):
                        comp_condition &= (
                                CPC_constr_list[n] in [technosphere_inputs_CPC[i] for i in id_technosphere_inputs_zero])

                    if comp_condition:
                        pass
                    else:
                        # if the construction phase was not detected via OWN_CONSTRUCTION,
                        # then we need the generic CONSTRUCTION CPC categories
                        CPC_inputs.extend(mapping_esm_flows_to_CPC_dict['CONSTRUCTION'])
                        ES_inputs.append('CONSTRUCTION')
                        set_CPC_inputs = set(CPC_inputs)
                        id_technosphere_inputs_zero = [i for i, e in enumerate(technosphere_inputs_CPC)
                                                       if e in set_CPC_inputs]

                for n in id_technosphere_inputs_zero:

                    flow = technosphere_inputs[n]

                    # Keep track of the amount in the original activity as a comment
                    old_amount = flow['amount']
                    flow['comment'] = f'Original amount: {old_amount}.' + flow.get('comment', '')
                    database = flow['database']
                    code = flow['code']
                    act_flow = self.main_database.db_as_dict_code[(database, code)]
                    res_categories = mapping_CPC_to_esm_flows_dict.get(dict(act_flow['classifications']).get('CPC', ''), '')

                    flows_set_to_zero.append([
                        tech,
                        new_act_op_d_c['reference product'],  # activity in which the flow is removed
                        new_act_op_d_c['name'],
                        new_act_op_d_c['location'],
                        new_act_op_d_c['database'],
                        new_act_op_d_c['code'],
                        old_amount, flow['unit'],  # quantity and unit
                        act_flow['reference product'],  # removed flow
                        act_flow['name'],
                        act_flow['location'],
                        act_flow['database'],
                        act_flow['code'],
                    ])

                    if create_new_db:
                        if 'OWN_CONSTRUCTION' in res_categories:
                            # replace construction flow input by the one added before in the ESM database
                            # (for validation purposes)
                            for idx, sub_comp in enumerate(self.technology_compositions_dict[tech]):
                                if code == act_constr_list[idx]['code']:
                                    new_code_constr = self.mapping_constr[self.mapping_constr.Name == sub_comp].New_code.iloc[0]
                                    flow['database'], flow['code'] = self.esm_db_name, new_code_constr

                    # add the removed amount in the ei_removal dict for post-analysis
                    for cat in res_categories:
                        if cat in ES_inputs:
                            # only adds the amount for the relevant category
                            # (e.g., ELECTRICITY_MV among [ELECTRICITY_LV, ELECTRICITY_MV, ELECTRICITY_HV]
                            # which share the same CPCs

                            # old amount (e.g., GWh) multiplied by factor as we went down in the tree
                            ei_removal[tech][cat]['amount'] += old_amount * new_act_op_d_c_amount
                            ei_removal[tech][cat]['count'] += 1  # count (i.e., number of flows put to zero)

                    # Setting the amount to zero
                    flow['amount'] = 0

                # Go deeper in the process tree if some flows were not found.
                # This is not applied to construction datasets, which should be found the foreground inventory.
                missing_ES_inputs = []
                for cat in ES_inputs:
                    if ((tech in list(self.background_search_act.keys()))
                            & (cat not in ['CONSTRUCTION', 'OWN_CONSTRUCTION', 'DECOMMISSIONING'])
                            & (ei_removal[tech][cat]['amount'] == 0) & (ei_removal[tech][cat]['count'] == 0)):
                        missing_ES_inputs.append(cat)

                if len(missing_ES_inputs) > 0:
                    if k_deep <= self.background_search_act[tech]:
                        perform_d_c = self.background_search(
                            act=new_act_op_d_c,
                            k=k_deep,
                            k_lim=self.background_search_act[tech] - 1,
                            amount=new_act_op_d_c_amount,
                            explore_type='background_removal',
                            ESM_inputs=missing_ES_inputs,
                            perform_d_c=perform_d_c,
                            create_new_db=create_new_db
                        )

                id_d_c += 1

        return flows_set_to_zero, ei_removal


def has_construction(row: pd.Series, no_construction_list: list[str]) -> int:
    """
    Add a construction input to technologies that have a construction phase

    :param row: row of the model file
    :param no_construction_list: list of technologies for which the construction phase is not considered
    :return: 0 if no construction phase, -1 otherwise
    """
    if row.Name in no_construction_list:
        return 0
    else:
        return -1


def has_decommissioning(row: pd.Series, decom_list: list[str]) -> int:
    """
    Add a decommissioning input to technologies that have a decommissioning phase outside their construction phase

    :param row: row of the model file
    :param decom_list: list of technologies for which the decommissioning phase is considered
    :return: -1 if decommissioning phase, 0 otherwise
    """
    if row.Name in decom_list:
        return -1
    else:
        return 0


def is_transport(row: pd.Series, mobility_list: list[str]) -> int:
    """
    Add a fuel input to mobility technologies (due to possible mismatch)

    :param row: row of the model file
    :param mobility_list: list of mobility technologies
    :return: -1 if mobility technology, 0 otherwise
    """
    if len(row[row == 1]) == 0:
        return 0
    elif row[row == 1].index[0] in mobility_list:
        return -1
    else:
        return 0


def is_process_activity(row: pd.Series, process_list: list[str]) -> int:
    """
    Add a fuel input to process activities that could have a mismatch

    :param row: row of the model file
    :param process_list: list of process activities
    :return: -1 if process activity, 0 otherwise
    """
    if row.Name in process_list:
        return -1
    else:
        return 0


def add_technology_specifics(mapping_op: pd.DataFrame, df_tech_specifics: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Add technology-specific inputs to the model file

    :param mapping_op: operation activities, mapping file merged with the model file
    :param df_tech_specifics: dataframe of technology specifics
    :return: updated mapping file
    """
    # Add a construction input to technologies that have a construction phase
    no_construction_list = list(df_tech_specifics[df_tech_specifics.Specifics == 'No construction'].Name)
    mapping_op['OWN_CONSTRUCTION'] = mapping_op.apply(lambda row: has_construction(row, no_construction_list), axis=1)

    # Add a decommissioning input to technologies that have a decommissioning phase outside their construction phase
    decom_list = list(df_tech_specifics[df_tech_specifics.Specifics == 'Decommissioning'].Name)
    mapping_op['DECOMMISSIONING'] = mapping_op.apply(lambda row: has_decommissioning(row, decom_list), axis=1)

    # Add a fuel input to mobility technologies (due to possible mismatch)
    mobility_list = list(df_tech_specifics[df_tech_specifics.Specifics == 'Mobility'].Name)
    mapping_op['TRANSPORT_FUEL'] = mapping_op.apply(lambda row: is_transport(row, mobility_list), axis=1)

    # Add a fuel input to process activities that could have a mismatch
    process_list = list(df_tech_specifics[df_tech_specifics.Specifics == 'Process'].Name)
    mapping_op['PROCESS_FUEL'] = mapping_op.apply(lambda row: is_process_activity(row, process_list), axis=1)

    return mapping_op
