import pandas as pd
import ast
import copy
import time
from pathlib import Path
from .modify_inventory import *
from .database import Database, Dataset
from .utils import random_code


class ESM:
    """
    Class that represents the ESM database, that can be modified with double-counting removal, regionalization,
    efficiency differences correction, and lifetime differences correction. LCA indicators can then be computed
    from it. And results from the ESM can be added back to the LCI database.
    """

    # Class variables
    best_loc_in_ranking = {}  # dictionary to store the best location for each activity

    def __init__(
            self,
            # Mandatory inputs
            mapping: pd.DataFrame,
            model: pd.DataFrame,
            unit_conversion: pd.DataFrame,
            mapping_esm_flows_to_CPC_cat: pd.DataFrame,
            main_database: Database,
            esm_db_name: str,

            # Optional inputs
            main_database_name: str = None,
            biosphere_db_name: str = None,
            technology_compositions: pd.DataFrame = None,
            results_path_file: str = 'results/',
            tech_specifics: pd.DataFrame = None,
            regionalize_foregrounds: bool = False,
            accepted_locations: list[str] = None,
            esm_location: str = None,
            locations_ranking: list[str] = None,
            spatialized_biosphere_db: Database = None,
            efficiency: pd.DataFrame = None,
            lifetime: pd.DataFrame = None,
    ):
        """
        Initialize the ESM database creation

        :param mapping: mapping between the ESM resources, technologies (operation and construction) and flows,
            and the LCI database activities
        :param model: dataframe containing the inputs and outputs of each technology in the ESM
        :param unit_conversion: dataframe containing unit conversion factors for all ESM technologies, resources and
            flows
        :param tech_specifics: dataframe containing the specific requirements (if any) of the ESM technologies
        :param technology_compositions: dataframe containing (if any) the compositions of technologies
        :param mapping_esm_flows_to_CPC_cat: mapping between ESM flows and CPC categories
        :param main_database: main LCI database, e.g., ecoinvent or premise database (with CPC categories)
        :param esm_db_name: name of the ESM database to be written in Brightway
        :param main_database_name: name of the main database (e.g., 'ecoinvent-3.9.1-cutoff') if main_database
            is an aggregation of the main database and complementary databases
        :param biosphere_db_name: name of the (not spatialized) biosphere database. Default is 'biosphere3'.
        :param results_path_file: path to your result folder
        :param regionalize_foregrounds: if True, regionalize the ESM database foreground
        :param accepted_locations: list of ecoinvent locations to keep without modification in case of regionalization
        :param esm_location: ecoinvent location corresponding to the geographical scope of the ESM
        :param locations_ranking: ranking of the preferred ecoinvent locations in case of regionalization
        :param spatialized_biosphere_db: spatialized biosphere database
        :param efficiency: dataframe containing the ESM technologies to correct regarding efficiency differences
            between the ESM and LCI database
        :param lifetime: dataframe containing the lifetime of the ESM technologies
        """

        self.mapping = mapping
        self.model = model
        self.tech_specifics = tech_specifics if tech_specifics is not None \
            else pd.DataFrame(columns=['Name', 'Specifics', 'Amount'])
        self.technology_compositions = technology_compositions if technology_compositions is not None \
            else pd.DataFrame(columns=['Name', 'Components'])
        self.mapping_esm_flows_to_CPC_cat = mapping_esm_flows_to_CPC_cat
        self.main_database = main_database
        self.main_database_name = main_database_name if main_database_name is not None else \
            (main_database.db_names if type(main_database.db_names) is str else main_database.db_names[0])
        self.biosphere_db_name = biosphere_db_name if biosphere_db_name is not None else 'biosphere3'
        self.esm_db_name = esm_db_name
        self.results_path_file = results_path_file
        self.regionalize_foregrounds = regionalize_foregrounds
        self.accepted_locations = accepted_locations
        self.esm_location = esm_location
        self.locations_ranking = locations_ranking
        self.spatialized_database = True if spatialized_biosphere_db is not None else False
        self.spatialized_biosphere_db = spatialized_biosphere_db
        self.efficiency = efficiency
        self.unit_conversion = unit_conversion
        self.lifetime = lifetime

    def __repr__(self):
        n_tech = self.mapping[(self.mapping['Type'] == 'Construction') | (self.mapping['Type'] == 'Operation')].shape[0]
        n_res = self.mapping[self.mapping['Type'] == 'Resource'].shape[0]
        return f"ESM Database with {n_tech} technologies and {n_res} resources"

    @property
    def mapping_op(self):
        mapping_op = self.mapping[self.mapping['Type'] == 'Operation']
        model_pivot = self.model.pivot(index='Name', columns='Flow', values='Amount').reset_index()
        model_pivot.fillna(0, inplace=True)
        mapping_op = pd.merge(mapping_op, model_pivot, on='Name', how='left')
        mapping_op['CONSTRUCTION'] = mapping_op.shape[0] * [0]
        mapping_op = self.add_technology_specifics(mapping_op)
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
            background_search_act[tech] = int(self.tech_specifics[self.tech_specifics.Name == tech].Amount.iloc[0])
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

    # Import methods from other files
    from .regionalization import (
        regionalize_activity_foreground,
        change_location_activity,
        change_location_mapping_file
    )
    from .double_counting import double_counting_removal, background_search
    from .impact_assessment import (
        compute_impact_scores,
        get_impact_categories,
        is_empty,
        aggregate_direct_emissions_activities
    )
    from .adapt_efficiency import (
        correct_esm_and_lca_efficiency_differences,
        compute_efficiency_esm,
        get_lca_input_flow_unit_or_product,
        adapt_biosphere_flows_to_efficiency_difference,
        get_lca_input_quantity,
    )
    from .esm_back_to_lca import (
        create_new_database_with_esm_results,
        create_or_modify_activity_from_esm_results,
        replace_mobility_end_use_type,
        connect_esm_results_to_database,
    )
    from .normalization import normalize_lca_metrics
    from .generate_lcia_obj_ampl import generate_mod_file_ampl

    def check_inputs(self) -> None:
        """
        Check if the inputs are consistent and send feedback to the user

        :return: None
        """
        # Check if the inputs are consistent
        model = self.model
        mapping = self.mapping
        mapping_esm_flows_to_CPC_cat = self.mapping_esm_flows_to_CPC_cat
        unit_conversion = self.unit_conversion
        efficiency = self.efficiency
        lifetime = self.lifetime
        techno_compositions = self.technology_compositions
        tech_specifics = self.tech_specifics

        dict_df_names = {
            'model': model,
            'mapping': mapping,
            'mapping_esm_flows_to_CPC_cat': mapping_esm_flows_to_CPC_cat,
            'unit_conversion': unit_conversion,
            'efficiency': efficiency,
            'lifetime': lifetime,
            'tech_specifics': tech_specifics,
            'technology_compositions': techno_compositions,
        }

        try:
            self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
        except ValueError:
            pass

        # Check for duplicates in all dataframes
        for df_name, df in dict_df_names.items():
            if df_name == 'technology_compositions':
                df['Components_tuple_temp'] = df.Components.apply(tuple)
                if df.duplicated(subset=['Name', 'Components_tuple_temp']).any():
                    print(f"Warning: there are duplicates in the {df_name} dataframe. Please check your inputs.")
                df.drop(columns=['Components_tuple_temp'], inplace=True)
            else:
                if df.duplicated().any():
                    print(f"Warning: there are duplicates in the {df_name} dataframe. Please check your inputs.")

        # Check if the technologies and resources in the model file are in the mapping file
        set_in_model_and_not_in_mapping = set()
        for tech_or_res in list(model.Name.unique()):
            if tech_or_res not in list(mapping[mapping.Type.isin(['Operation', 'Construction', 'Resource'])].Name):
                set_in_model_and_not_in_mapping.add(tech_or_res)
        if len(set_in_model_and_not_in_mapping) > 0:
            print(f"List of technologies or resources that are in the model file but not in the mapping file. "
                  f"Their impact scores will be set to the default value.\n")
            print(f"--> {sorted(set_in_model_and_not_in_mapping)}\n")

        # Check if the technologies and resources in the mapping file are in the model file
        set_in_mapping_and_not_in_model = set()
        list_subcomponents = [x for xs in list(techno_compositions.Components) for x in xs]
        for tech_or_res in list(mapping[mapping.Type.isin(['Operation', 'Construction', 'Resource'])].Name):
            if tech_or_res not in list(model.Name.unique()):
                if tech_or_res in list_subcomponents:
                    pass
                else:
                    set_in_mapping_and_not_in_model.add(tech_or_res)
        if len(set_in_mapping_and_not_in_model) > 0:
            print(f"List of technologies or resources that are in the mapping file but not in the model file "
                  f"(this will not be a problem in the workflow).\n")
            print(f"--> {sorted(set_in_mapping_and_not_in_model)}\n")

        # Check if the technologies and resources in the mapping file are in the unit conversion file
        set_in_mapping_and_not_in_unit_conversion = set()
        for tech_or_res in list(mapping[mapping.Type.isin(['Operation', 'Construction', 'Resource'])].Name):
            if tech_or_res not in list(unit_conversion.Name):
                set_in_mapping_and_not_in_unit_conversion.add(tech_or_res)
        if len(set_in_mapping_and_not_in_unit_conversion) > 0:
            print(f"List of technologies or resources that are in the mapping file but not in the unit conversion file. "
                  f"It might be an issue if unit conversions are required during the impact assessment step.\n")
            print(f"--> {sorted(set_in_mapping_and_not_in_unit_conversion)}\n")

        # Check if the flows in the model file are in the ESM flows - CPC mapping file
        set_flows_not_in_mapping_esm_flows_to_CPC_cat = set()
        for flow in list(model.Flow.unique()):
            if ((flow not in list(mapping_esm_flows_to_CPC_cat.Flow))  # Flow not in the mapping file
                    & (len(model[(model.Flow == flow) & (model.Amount < 0)]) > 0)):  # Flow used as an input
                set_flows_not_in_mapping_esm_flows_to_CPC_cat.add(flow)
        if len(set_flows_not_in_mapping_esm_flows_to_CPC_cat) > 0:
            print(f"List of flows that are in the model file but not in the ESM flows to CPC mapping file. "
                  f"It might be an issue for double counting if these flows are inputs of SOME esm technologies. \n")
            print(f"--> {sorted(set_flows_not_in_mapping_esm_flows_to_CPC_cat)}\n")

        if lifetime is not None:
            # Check if the technologies in the mapping file are in the lifetime file
            set_in_mapping_and_not_in_lifetime = set()
            for tech in list(mapping[mapping.Type == 'Construction'].Name):
                if tech not in list(lifetime.Name):
                    set_in_mapping_and_not_in_lifetime.add(tech)
            if len(set_in_mapping_and_not_in_lifetime) > 0:
                print(f"List of technologies that are in the mapping file but not in the lifetime file. "
                      f"Please add the missing technologies or remove the lifetime file.\n")
                print(f"--> {sorted(set_in_mapping_and_not_in_lifetime)}\n")

        if efficiency is not None:
            # Check if the technologies in the efficiency file are in the mapping file and the model file
            set_in_efficiency_and_not_in_mapping = set()
            for tech in list(efficiency.Name):
                if tech not in list(mapping[mapping.Type == 'Operation'].Name):
                    set_in_efficiency_and_not_in_mapping.add(tech)
            if len(set_in_efficiency_and_not_in_mapping) > 0:
                print(f"List of technologies that are in the efficiency file but not in the mapping file "
                      f"(this will not be a problem in the workflow).\n")
                print(f"--> {sorted(set_in_efficiency_and_not_in_mapping)}\n")

            set_in_efficiency_and_not_in_model = set()
            for tech in list(efficiency.Name):
                if tech not in list(model.Name):
                    set_in_efficiency_and_not_in_model.add(tech)
            if len(set_in_efficiency_and_not_in_model) > 0:
                print(f"List of technologies that are in the efficiency file but not in the model file. You should "
                      f"remove these technologies from the efficiency file, as the efficiency in the model cannot be "
                      f"retrieved.\n")
                print(f"--> {sorted(set_in_efficiency_and_not_in_model)}\n")

        # Check if the technologies in the tech_specifics file are in the mapping file
        set_in_tech_specifics_and_not_in_mapping = set()
        for tech in list(tech_specifics.Name):
            if tech not in list(mapping.Name):
                set_in_tech_specifics_and_not_in_mapping.add(tech)
        if len(set_in_tech_specifics_and_not_in_mapping) > 0:
            print(f"List of technologies that are in the tech_specifics file but not in the mapping file "
                  f"(this will not be a problem in the workflow).\n")
            print(f"--> {sorted(set_in_tech_specifics_and_not_in_mapping)}\n")

        # Check that sub-technologies in the technology_compositions file are in the mapping file
        set_sub_techs_not_in_mapping = [
            sub_tech for sub_tech_list in self.technology_compositions.Components for sub_tech in sub_tech_list
            if sub_tech not in list(mapping[mapping.Type == 'Construction'].Name.unique())
        ]
        if len(set_sub_techs_not_in_mapping) > 0:
            set_sub_techs_not_in_mapping = set(set_sub_techs_not_in_mapping)
            print(f"List of sub-technologies that are in the technology_compositions file but not in the mapping file "
                  f"(this will not be a problem in the workflow).\n")
            print(f"--> {sorted(set_sub_techs_not_in_mapping)}\n")

    def create_esm_database(
            self,
            return_database: bool = False,
            write_database: bool = True,
            write_double_counting_removal_reports: bool = True,
    ) -> Database | None:
        """
        Create the ESM database after double counting removal. Three csv files summarizing the double-counting removal
        process are automatically saved in the results folder: double_counting_removal.csv (amount of removed
        flows), double_counting_removal_count.csv (number of flows set to zero), and removed_flows_list.csv
        (specific activities in which the flows were removed).

        :param return_database: if True, return the ESM database as a mescal.Database object
        :param write_database: if True, write the ESM database to Brightway
        :param write_double_counting_removal_reports: if True, write the double-counting removal reports in the results
            folder
        :return: the ESM database if return_database is True, None otherwise
        """

        try:
            self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
        except ValueError:
            pass

        if (self.efficiency is not None) & (self.unit_conversion is None):
            raise ValueError('Unit conversion file is needed for efficiency differences correction. Please provide it.')

        if write_database is False and return_database is False:
            raise ValueError('Please set either return_database or write_database to True.')

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
        print(f"### Starting to add construction and resource activities database ###")
        t1_add = time.time()
        self.add_activities_to_database(act_type='Construction')
        self.add_activities_to_database(act_type='Resource')
        t2_add = time.time()
        print(f"### Construction and resource activities added to the database ###")
        print(f"--> Time: {round(t2_add - t1_add, 0)} seconds")

        print(f"### Starting to remove double-counted flows ###")
        t1_dc = time.time()
        (
            flows_set_to_zero,
            ei_removal,
            activities_subject_to_double_counting
        ) = self.double_counting_removal(df_op=self.mapping_op, N=N, ESM_inputs='all')
        t2_dc = time.time()
        print(f"### Double-counting removal done ###")
        print(f"--> Time: {round(t2_dc - t1_dc, 0)} seconds")

        df_flows_set_to_zero = pd.DataFrame(
            data=flows_set_to_zero,
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
            print(f"### Starting to correct efficiency differences ###")
            t1_eff = time.time()
            if self.efficiency is not None:
                self.correct_esm_and_lca_efficiency_differences(
                    removed_flows=df_flows_set_to_zero,
                    double_counting_removal=double_counting_removal_amount,
                )
            t2_eff = time.time()
            print(f"### Efficiency differences corrected ###")
            print(f"--> Time: {round(t2_eff - t1_eff, 0)} seconds")

        if write_double_counting_removal_reports:
            Path(self.results_path_file).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
            double_counting_removal_amount.to_csv(f"{self.results_path_file}double_counting_removal.csv", index=False)
            double_counting_removal_count.to_csv(f"{self.results_path_file}double_counting_removal_count.csv",
                                                 index=False)
            df_flows_set_to_zero.to_csv(f"{self.results_path_file}removed_flows_list.csv", index=False)
            pd.DataFrame(
                data=activities_subject_to_double_counting,
                columns=['Name', 'Activity name', 'Activity code', 'Amount']
            ).to_csv(f"{self.results_path_file}activities_subject_to_double_counting.csv", index=False)

        if write_database:
            print(f"### Starting to write database ###")
            t1_mod_inv = time.time()

            esm_db = Database(
                db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])
            esm_db.write_to_brightway(self.esm_db_name)

            # Modify the written database according to the tech_specifics.csv file
            self.modify_written_activities(db=esm_db)

            t2_mod_inv = time.time()
            print("### Database written ###")
            print(f"--> Time: {round(t2_mod_inv - t1_mod_inv, 0)} seconds")

        if return_database:
            esm_db = Database(
                db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])

            self.main_database = self.main_database - esm_db  # Remove ESM database from the main database
            return esm_db

        self.main_database = self.main_database - Database(
            db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])

    def add_technology_specifics(
            self,
            mapping_op: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add technology-specific inputs to the model file

        :param mapping_op: operation activities, mapping file merged with the model file
        :return: the updated mapping file
        """

        df_tech_specifics = self.tech_specifics

        # Add a construction input to technologies that have a construction phase
        no_construction_list = list(df_tech_specifics[df_tech_specifics.Specifics == 'No construction'].Name)
        mapping_op['OWN_CONSTRUCTION'] = mapping_op.apply(lambda row: has_construction(row, no_construction_list),
                                                          axis=1)

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

    def add_activities_to_database(
            self,
            act_type: str,
    ) -> None:
        """
        Add new activities to the main database

        :param act_type: the type of activity, it can be 'Construction', 'Operation', or 'Resource'
        :return: None
        """

        mapping_type = self.mapping[self.mapping['Type'] == act_type]
        db_as_list = self.main_database.db_as_list
        db_as_dict_code = self.main_database.db_as_dict_code

        for i in range(len(mapping_type)):
            ds = self.create_new_activity(
                name=mapping_type['Name'].iloc[i],
                act_type=act_type,
                current_code=mapping_type['Current_code'].iloc[i],
                new_code=mapping_type['New_code'].iloc[i],
                database_name=mapping_type['Database'].iloc[i],
                db_as_dict_code=db_as_dict_code,
            )
            db_as_list.append(ds)
        self.main_database.db_as_list = db_as_list

    def create_new_activity(
            self,
            name: str,
            act_type: str,
            current_code: str,
            new_code: str,
            database_name: str,
            db_as_dict_code: dict,
    ) -> dict:
        """
        Create a new LCI dataset for the ESM technology or resource

        :param name: name of the technology or resource in the ESM
        :param act_type: the type of activity, it can be 'Construction', 'Operation', or 'Resource'
        :param current_code: code of the activity in the original LCI database
        :param new_code: code of the new activity in the new LCI database
        :param database_name: name of the original LCI database
        :param db_as_dict_code: dictionary of the original LCI database with (database, code) as key
        :return: the new LCI dataset for the technology or resource
        """

        act = db_as_dict_code[(database_name, current_code)]

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

    def modify_written_activities(
            self,
            db: Database,
            db_type: str = 'esm',
    ) -> None:
        """
        Modify the written database according to the tech_specifics.csv file and using functions from
        modify_inventory.py

        :param db: LCI database
        :param db_type: type of LCI database, can be 'esm', 'esm results' or 'main'
        :return: None (activities are modified in the brightway project)
        """
        biosphere_db_name = self.biosphere_db_name

        if db_type == 'esm':
            db_name = self.esm_db_name
            return_type = 'name'
        elif db_type == 'esm results':
            db_name = self.esm_db_name + '_results'
            return_type = 'code'
        elif db_type == 'main':
            db_name = db.db_names
            return_type = 'code'
        else:
            raise ValueError('db_type must be either "esm", "esm results" or "main"')

        # Change carbon flow of DAC from biogenic to fossil
        dac_technologies = list(self.tech_specifics[self.tech_specifics.Specifics == 'DAC'].Name)
        for tech in dac_technologies:
            activity_name_or_code = self.get_activity_name_or_code(tech=tech, return_type=return_type)
            if activity_name_or_code in [act[return_type] for act in db.db_as_list]:
                if return_type == 'name':
                    change_dac_biogenic_carbon_flow(
                        db_name=db_name,
                        activity_name=activity_name_or_code,
                        biosphere_db_name=biosphere_db_name,
                    )
                elif return_type == 'code':
                    change_dac_biogenic_carbon_flow(
                        db_name=db_name,
                        activity_code=activity_name_or_code,
                        biosphere_db_name=biosphere_db_name,
                    )

        # Change carbon flows of biofuel mobility technologies
        biofuel_mob_tech = self.tech_specifics[self.tech_specifics.Specifics == 'Biofuel'][
            ['Name', 'Amount']].values.tolist()
        for tech, biogenic_ratio in biofuel_mob_tech:
            activity_name_or_code = self.get_activity_name_or_code(tech=tech, return_type=return_type)
            if activity_name_or_code in [act[return_type] for act in db.db_as_list]:
                if return_type == 'name':
                    change_fossil_carbon_flows_of_biofuels(
                        db_name=db_name,
                        activity_name=activity_name_or_code,
                        biogenic_ratio=float(biogenic_ratio),
                        biosphere_db_name=biosphere_db_name,
                    )
                elif return_type == 'code':
                    change_fossil_carbon_flows_of_biofuels(
                        db_name=db_name,
                        activity_code=activity_name_or_code,
                        biogenic_ratio=float(biogenic_ratio),
                        biosphere_db_name=biosphere_db_name,
                    )

        # Adjust carbon flows by a constant factor for some technologies
        carbon_flows_correction_tech = self.tech_specifics[self.tech_specifics.Specifics == 'Carbon flows'][
            ['Name', 'Amount']].values.tolist()
        for tech, factor in carbon_flows_correction_tech:
            activity_name_or_code = self.get_activity_name_or_code(tech=tech, return_type=return_type)
            if activity_name_or_code in [act[return_type] for act in db.db_as_list]:
                if return_type == 'name':
                    change_direct_carbon_emissions_by_factor(
                        db_name=db_name,
                        activity_name=activity_name_or_code,
                        factor=float(factor),
                    )
                elif return_type == 'code':
                    change_direct_carbon_emissions_by_factor(
                        db_name=db_name,
                        activity_code=activity_name_or_code,
                        factor=float(factor),
                    )

        # Add carbon capture to plant
        add_carbon_capture_tech = self.tech_specifics[self.tech_specifics.Specifics == 'Add CC'][
            ['Name', 'Amount']].values.tolist()
        for tech, type_and_ratio in add_carbon_capture_tech:
            activity_name_or_code = self.get_activity_name_or_code(tech=tech, return_type=return_type)
            if activity_name_or_code in [act[return_type] for act in db.db_as_list]:
                if return_type == 'name':
                    type_and_ratio = type_and_ratio.split(', ')
                    add_carbon_capture_to_plant(
                        activity_database_name=db_name,
                        premise_database_name=self.main_database_name,
                        activity_name=activity_name_or_code,
                        plant_type=str(type_and_ratio[0]),
                        capture_ratio=float(type_and_ratio[1]),
                    )
                elif return_type == 'code':
                    type_and_ratio = type_and_ratio.split(', ')
                    add_carbon_capture_to_plant(
                        activity_database_name=db_name,
                        premise_database_name=self.main_database_name,
                        activity_code=activity_name_or_code,
                        plant_type=str(type_and_ratio[0]),
                        capture_ratio=float(type_and_ratio[1]),
                    )

    def get_activity_name_or_code(
            self,
            tech: str,
            return_type: str,
            phase: str = 'Operation',
    ) -> str:
        """
        Returns the name of code of the activity

        :param tech: name of the ESM technology
        :param return_type: type of return, can be 'name' or 'code'
        :param phase: phase of the technology, can be 'Operation' or 'Construction'
        :return: name or code
        """
        if return_type == 'name':
            return f'{tech}, {phase}'
        elif return_type == 'code':
            return self.mapping[(self.mapping.Name == tech) & (self.mapping.Type == phase)].New_code.iloc[0]


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
