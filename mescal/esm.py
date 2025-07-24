import pandas as pd
import ast
import copy
import time
import logging
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
            esm_location: str = 'GLO',
            locations_ranking: list[str] = None,
            spatialized_biosphere_db: Database = None,
            efficiency: pd.DataFrame = None,
            lifetime: pd.DataFrame = None,
            max_depth_double_counting_search: int = 10,
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
        :param max_depth_double_counting_search: maximum recursion depth of the double-counting background search
            algorithm
        """

        # set up logging tool
        self.logger = logging.getLogger('Mescal')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

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
        self.accepted_locations = accepted_locations if accepted_locations is not None else [esm_location]
        self.esm_location = esm_location
        self.locations_ranking = locations_ranking
        self.spatialized_database = True if spatialized_biosphere_db is not None else False
        self.spatialized_biosphere_db = spatialized_biosphere_db
        self.efficiency = efficiency
        self.unit_conversion = unit_conversion
        self.lifetime = lifetime
        self.max_depth_double_counting_search = max_depth_double_counting_search

        # Initialize attributes used within mescal
        self.df_flows_set_to_zero = None
        self.double_counting_removal_amount = None
        self.double_counting_removal_count = None
        self.df_activities_subject_to_double_counting = None
        self.esm_results_db_name = self.esm_db_name + '_results'
        self.pathway = False
        self.operation_metrics_for_all_time_steps = False
        self.year = None
        self.list_of_years = [None]
        self.esm_db = None
        self.tech_to_remove_layers = None
        self.efficiency_differences_report = None
        self.products_without_a_cpc_category = set()


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
        mapping_op = self._add_technology_specifics(mapping_op)
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
        _regionalize_activity_foreground,
        _change_location_activity,
        change_location_mapping_file
    )
    from .double_counting import _double_counting_removal, _background_search
    from .impact_assessment import (
        compute_impact_scores,
        _get_impact_categories,
        _is_empty,
        _aggregate_direct_emissions_activities
    )
    from .adapt_efficiency import (
        _correct_esm_and_lca_efficiency_differences,
        _compute_efficiency_esm,
        _get_lca_input_flow_unit_or_product,
        _adapt_flows_to_efficiency_difference,
        _get_lca_input_quantity,
    )
    from .esm_back_to_lca import (
        create_new_database_with_esm_results,
        _create_or_modify_activity_from_esm_results,
        _replace_mobility_end_use_type,
        connect_esm_results_to_database,
        _correct_esm_and_lca_capacity_factor_differences,
    )
    from .normalization import normalize_lca_metrics, _tech_type
    from .generate_lcia_obj_ampl import generate_mod_file_ampl

    def check_inputs(self) -> None:
        """
        Check if the inputs are consistent and send feedback to the user

        :return: None
        """
        # Check if the inputs are consistent
        main_database_name = self.main_database_name
        biosphere_db_name = self.biosphere_db_name
        model = self.model
        mapping = self.mapping
        mapping_esm_flows_to_CPC_cat = self.mapping_esm_flows_to_CPC_cat
        unit_conversion = self.unit_conversion
        efficiency = self.efficiency
        lifetime = self.lifetime
        techno_compositions = self.technology_compositions
        tech_specifics = self.tech_specifics

        if main_database_name not in list(bd.databases):
            self.logger.error(f"Main database {main_database_name} not found in your brightway project")

        if biosphere_db_name not in list(bd.databases):
            self.logger.error(f"Biosphere database {biosphere_db_name} not found in your brightway project")

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
                    self.logger.warning(f"There are duplicates in the {df_name} dataframe. Please check your inputs.")
                df.drop(columns=['Components_tuple_temp'], inplace=True)
            else:
                if df.duplicated().any():
                    self.logger.warning(f"There are duplicates in the {df_name} dataframe. Please check your inputs.")

        # Check if the technologies and resources in the model file are in the mapping file
        set_in_model_and_not_in_mapping = set()
        for tech_or_res in list(model.Name.unique()):
            if tech_or_res not in list(mapping[mapping.Type.isin(['Operation', 'Construction', 'Resource'])].Name):
                set_in_model_and_not_in_mapping.add(tech_or_res)
        if len(set_in_model_and_not_in_mapping) > 0:
            self.logger.warning(
                f"List of technologies or resources that are in the model file but not in the mapping file. "
                f"Their impact scores will be set to the default value: {sorted(set_in_model_and_not_in_mapping)}"
            )

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
            self.logger.warning(
                f"List of technologies or resources that are in the mapping file but not in the model file "
                f"(this will not be a problem in the workflow): {sorted(set_in_mapping_and_not_in_model)}"
            )

        # Check if the technologies and resources in the mapping file are in the unit conversion file
        set_in_mapping_and_not_in_unit_conversion = set()
        for tech_or_res in list(mapping[mapping.Type.isin(['Operation', 'Construction', 'Resource'])].Name):
            if tech_or_res not in list(unit_conversion.Name):
                set_in_mapping_and_not_in_unit_conversion.add(tech_or_res)
        if len(set_in_mapping_and_not_in_unit_conversion) > 0:
            self.logger.warning(
                f"List of technologies or resources that are in the mapping file but not in the unit conversion file. "
                f"It might be an issue if unit conversions are required during the impact assessment step: "
                f"{sorted(set_in_mapping_and_not_in_unit_conversion)}"
            )

        # Check if the flows in the model file are in the ESM flows - CPC mapping file
        set_flows_not_in_mapping_esm_flows_to_CPC_cat = set()
        for flow in list(model.Flow.unique()):
            if ((flow not in list(mapping_esm_flows_to_CPC_cat.Flow))  # Flow not in the mapping file
                    & (len(model[(model.Flow == flow) & (model.Amount < 0)]) > 0)):  # Flow used as an input
                set_flows_not_in_mapping_esm_flows_to_CPC_cat.add(flow)
        if len(set_flows_not_in_mapping_esm_flows_to_CPC_cat) > 0:
            self.logger.warning(
                f"List of flows that are in the model file but not in the ESM flows to CPC mapping file. "
                f"It might be an issue for double counting if these flows are inputs of some ESM technologies: "
                f"{sorted(set_flows_not_in_mapping_esm_flows_to_CPC_cat)}"
            )

        if lifetime is not None:
            # Check if the technologies in the mapping file are in the lifetime file
            set_in_mapping_and_not_in_lifetime = set()
            for tech in list(mapping[mapping.Type == 'Construction'].Name):
                if tech not in list(lifetime.Name):
                    set_in_mapping_and_not_in_lifetime.add(tech)
            if len(set_in_mapping_and_not_in_lifetime) > 0:
                self.logger.warning(
                    f"List of technologies that are in the mapping file but not in the lifetime file. "
                    f"Please add the missing technologies or remove the lifetime file: "
                    f"{sorted(set_in_mapping_and_not_in_lifetime)}"
                )

        if efficiency is not None:
            # Check if the technologies in the efficiency file are in the mapping file and the model file
            set_in_efficiency_and_not_in_mapping = set()
            for tech in list(efficiency.Name):
                if tech not in list(mapping[mapping.Type == 'Operation'].Name):
                    set_in_efficiency_and_not_in_mapping.add(tech)
            if len(set_in_efficiency_and_not_in_mapping) > 0:
                self.logger.warning(
                    f"List of technologies that are in the efficiency file but not in the mapping file "
                    f"(this will not be a problem in the workflow): {sorted(set_in_efficiency_and_not_in_mapping)}"
                )

            set_in_efficiency_and_not_in_model = set()
            for tech in list(efficiency.Name):
                if tech not in list(model.Name):
                    set_in_efficiency_and_not_in_model.add(tech)
            if len(set_in_efficiency_and_not_in_model) > 0:
                self.logger.warning(
                    f"List of technologies that are in the efficiency file but not in the model file. You should "
                    f"remove these technologies from the efficiency file, as the efficiency in the model cannot be "
                    f"retrieved: {sorted(set_in_efficiency_and_not_in_model)}"
                )

        # Check if the technologies in the tech_specifics file are in the mapping file
        set_in_tech_specifics_and_not_in_mapping = set()
        for tech in list(tech_specifics.Name):
            if tech not in list(mapping.Name):
                set_in_tech_specifics_and_not_in_mapping.add(tech)
        if len(set_in_tech_specifics_and_not_in_mapping) > 0:
            self.logger.warning(
                f"List of technologies that are in the tech_specifics file but not in the mapping file "
                f"(this will not be a problem in the workflow): {sorted(set_in_tech_specifics_and_not_in_mapping)}"
            )

        # Check that sub-technologies in the technology_compositions file are in the mapping file
        set_sub_techs_not_in_mapping = [
            sub_tech for sub_tech_list in self.technology_compositions.Components for sub_tech in sub_tech_list
            if sub_tech not in list(mapping[mapping.Type == 'Construction'].Name.unique())
        ]
        if len(set_sub_techs_not_in_mapping) > 0:
            set_sub_techs_not_in_mapping = set(set_sub_techs_not_in_mapping)
            self.logger.warning(
                f"List of sub-technologies that are in the technology_compositions file but not in the mapping file "
                f"(this will not be a problem in the workflow): {sorted(set_sub_techs_not_in_mapping)}"
            )

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

        if write_database is False and len(self.tech_specifics) > 0:
            raise ValueError('The database must be written to Brightway to apply technologies specifics. '
                             'Please set write_database to True.')

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
        self.logger.info("Starting to add construction and resource activities database")
        t1_add = time.time()
        self._add_activities_to_database(act_type='Construction')
        self._add_activities_to_database(act_type='Resource')
        t2_add = time.time()
        self.logger.info(f"Construction and resource activities added to the database in {round(t2_add - t1_add, 1)} "
                         f"seconds")

        self.logger.info("Starting to remove double-counted flows")
        t1_dc = time.time()
        (
            flows_set_to_zero,
            ei_removal,
            activities_subject_to_double_counting
        ) = self._double_counting_removal(df_op=self.mapping_op, N=N, ESM_inputs='all')
        t2_dc = time.time()
        self.logger.info(f"Double-counting removal done in {round(t2_dc - t1_dc, 1)} seconds")

        if len(self.products_without_a_cpc_category) > 0:
            self.logger.error(
                f'Some products in your foreground inventory do not have a CPC category, please map them in a '
                f'mapping_new_products_to_CPC dataframe, and give the latter as an argument of the add_CPC_categories '
                f'method of the Database class. Here is the list of products without a CPC category: '
                f'{self.products_without_a_cpc_category}'
            )

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

        df_activities_subject_to_double_counting = pd.DataFrame(
            data=activities_subject_to_double_counting,
            columns=['Name', 'Activity name', 'Activity code', 'Amount']
        )

        self.double_counting_removal_count = double_counting_removal_count
        self.double_counting_removal_amount = double_counting_removal_amount
        self.df_flows_set_to_zero = df_flows_set_to_zero
        self.df_activities_subject_to_double_counting = df_activities_subject_to_double_counting

        if self.efficiency is not None:
            self.logger.info(f"Starting to correct efficiency differences")
            t1_eff = time.time()
            if self.efficiency is not None:
                self._correct_esm_and_lca_efficiency_differences()
            t2_eff = time.time()
            self.logger.info(f"Efficiency differences corrected in {round(t2_eff - t1_eff, 1)} seconds")

        if write_double_counting_removal_reports:
            Path(self.results_path_file).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
            double_counting_removal_amount.to_csv(f"{self.results_path_file}double_counting_removal.csv", index=False)
            double_counting_removal_count.to_csv(f"{self.results_path_file}double_counting_removal_count.csv",
                                                 index=False)
            df_flows_set_to_zero.to_csv(f"{self.results_path_file}removed_flows_list.csv", index=False)
            df_activities_subject_to_double_counting.to_csv(f"{self.results_path_file}activities_subject_to_double_counting.csv", index=False)

        if write_database:
            self.logger.info(f"Starting to write database")
            t1_mod_inv = time.time()

            self.esm_db = Database(
                db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])
            self.esm_db.write_to_brightway(self.esm_db_name)

            # Modify the written database according to the tech_specifics.csv file
            self._modify_written_activities(db=self.esm_db)

            t2_mod_inv = time.time()
            self.logger.info(f"Database written in {round(t2_mod_inv - t1_mod_inv, 1)} seconds")

        if return_database:
            if write_database:
                self.esm_db = Database(db_names=self.esm_db_name)  # accounts for modifications from tech_specifics.csv file
            else:
                self.esm_db = Database(
                    db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])

            self.main_database = self.main_database - self.esm_db  # Remove ESM database from the main database
            return self.esm_db

        self.main_database = self.main_database - Database(
            db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])

    def _add_technology_specifics(
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

    def _add_activities_to_database(
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
            ds = self._create_new_activity(
                name=mapping_type['Name'].iloc[i],
                act_type=act_type,
                current_code=mapping_type['Current_code'].iloc[i],
                new_code=mapping_type['New_code'].iloc[i],
                database_name=mapping_type['Database'].iloc[i],
                db_as_dict_code=db_as_dict_code,
            )
            db_as_list.append(ds)
        self.main_database.db_as_list = db_as_list

    def _create_new_activity(
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
            new_act = self._regionalize_activity_foreground(act=new_act)

        return new_act

    def _modify_written_activities(
            self,
            db: Database,
            db_type: str = 'esm',
    ) -> None:
        """
        Modify the written database according to the tech_specifics.csv file and using functions from
        modify_inventory.py

        :param db: LCI database
        :param db_type: type of LCI database can be 'esm', 'esm results' or 'main'
        :return: None (activities are modified in the brightway project)
        """
        biosphere_db_name = self.biosphere_db_name

        if db_type == 'esm':
            db_name = self.esm_db_name
            return_type = 'name'
        elif db_type == 'esm results':
            db_name = self.esm_results_db_name
            return_type = 'code'
        elif db_type == 'main':
            db_name = db.db_names
            return_type = 'code'
        else:
            raise ValueError('db_type must be either "esm", "esm results" or "main"')

        # Change carbon flow of DAC from biogenic to fossil
        dac_technologies = list(self.tech_specifics[self.tech_specifics.Specifics == 'DAC'].Name)
        for tech in dac_technologies:
            activity_name_or_code = self._get_activity_name_or_code(tech=tech, return_type=return_type)
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
            activity_name_or_code = self._get_activity_name_or_code(tech=tech, return_type=return_type)
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
            activity_name_or_code = self._get_activity_name_or_code(tech=tech, return_type=return_type)
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
            activity_name_or_code = self._get_activity_name_or_code(tech=tech, return_type=return_type)
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

    def _get_activity_name_or_code(
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

    def _get_original_code(self) -> None:
        """
        Creates the Current_code column in the mapping DataFrame, which contains the original code from the main database.

        :return: None (updates the mapping DataFrame)
        """
        main_db_as_dict_name = self.main_database.db_as_dict_name
        self.mapping['Current_code'] = self.mapping.apply(lambda x: main_db_as_dict_name[(
            x['Activity'],
            x['Product'],
            x['Location'],
            x['Database'],
        )]['code'], axis=1)

    def _get_new_code(self) -> None:
        """
        Creates the New_code column in the mapping DataFrame, which contains the new code from the ESM database.

        :return: None (updates the mapping DataFrame)
        """
        esm_db_name = self.esm_db_name
        if self.esm_db is not None:
            esm_db = self.esm_db
        else:
            esm_db = Database(esm_db_name)
        esm_db_as_dict_name = esm_db.db_as_dict_name
        if self.operation_metrics_for_all_time_steps:
            self.mapping['New_code'] = self.mapping.apply(
                lambda x: self._get_new_code_previous_years(x, esm_db_as_dict_name)
                if x['Type'] in ['Construction', 'Operation', 'Resource'] else None, axis=1)
        else:
            self.mapping['New_code'] = self.mapping.apply(
                lambda x: self._get_new_code_iteration(x, esm_db_as_dict_name)
                if x['Type'] in ['Construction', 'Operation', 'Resource'] else None, axis=1)

    def _get_new_code_iteration(self, row: pd.Series, esm_db_as_dict_name: dict) -> str:
        """
        Function to iterate over the rows of the mapping DataFrame and get the new code for each activity.

        :param row: row of the mapping DataFrame
        :param esm_db_as_dict_name: dictionary of the ESM database with (name, product, location, database) as key
        :return: code of the activity in the ESM database
        """
        if self.regionalize_foregrounds:
            try:
                return esm_db_as_dict_name[(
                    f"{row['Name']}, {row['Type']}",
                    row['Product'],
                    self.esm_location,
                    self.esm_db_name,
                )]['code']
            except KeyError:
                return esm_db_as_dict_name[(
                    f"{row['Name']}, {row['Type']}",
                    row['Product'],
                    row['Location'],
                    self.esm_db_name,
                )]['code']
        else:
            return self.esm_db.db_as_dict_name[(
                f"{row['Name']}, {row['Type']}",
                row['Product'],
                row['Location'],
                self.esm_db_name,
            )]['code']

    def _get_new_code_previous_years(self, row: pd.Series, esm_db_as_dict_name: dict) -> str:
        """
        Function to iterate over the rows of the mapping DataFrame and get the new code for each activity,
        considering the year of the activity. This is used when operation metrics for all time steps are required.

        :param row: row of the mapping DataFrame
        :param esm_db_as_dict_name: dictionary of the ESM database with (name, product, location, database) as key
        :return: code of the activity in the ESM database
        """
        if self.regionalize_foregrounds:
            try:
                if row['Year'] == self.year:
                    return esm_db_as_dict_name[(
                        f"{row['Name']}, {row['Type']}",
                        row['Product'],
                        self.esm_location,
                        self.esm_db_name,
                    )]['code']
                elif row['Year'] < self.year:
                    return esm_db_as_dict_name[(
                        f"{row['Name']}, {row['Type']} ({row['Year']})",
                        row['Product'],
                        self.esm_location,
                        self.esm_db_name,
                    )]['code']
                else:
                    raise ValueError(f"Year of the following row is greater than the current year {self.year}: {row}")
            except KeyError:
                if row['Year'] == self.year:
                    return esm_db_as_dict_name[(
                        f"{row['Name']}, {row['Type']}",
                        row['Product'],
                        row['Location'],
                        self.esm_db_name,
                    )]['code']
                elif row['Year'] < self.year:
                    return esm_db_as_dict_name[(
                        f"{row['Name']}, {row['Type']} ({row['Year']})",
                        row['Product'],
                        row['Location'],
                        self.esm_db_name,
                    )]['code']
                else:
                    raise ValueError(f"Year of the following row is greater than the current year {self.year}: {row}")
        else:
            if row['Year'] == self.year:
                return esm_db_as_dict_name[(
                    f"{row['Name']}, {row['Type']}",
                    row['Product'],
                    row['Location'],
                    self.esm_db_name,
                )]['code']
            elif row['Year'] < self.year:
                return esm_db_as_dict_name[(
                    f"{row['Name']}, {row['Type']} ({row['Year']})",
                    row['Product'],
                    row['Location'],
                    self.esm_db_name,
                )]['code']
            else:
                raise ValueError(f"Year of the following row is greater than the current year {self.year}: {row}")

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

class PathwayESM(ESM):
    """
    The PathwayESM class inherits from the ESM class and is used to create the ESM databases, impact score
    dataframes, .dat files, etc. corresponding to all time steps of a pathway ESM.
    """

    def __init__(
            self,
            time_steps: list[dict],
            operation_metrics_for_all_time_steps: bool = False,
            *args, **kwargs
    ):
        """
        Initialize the PathwayESM class. See ESM.__init__ for full argument documentation.

        :param time_steps: List of dictionaries, each containing parameters for a time step in the pathway ESM.
            A time step should contain at least the 'year' and 'main_database' keys, and optionally
            'main_database_name', 'model' and 'lifetime'.
        :param operation_metrics_for_all_time_steps: if True, the operation metrics for technologies that were
            installed in previous time steps (i.e., with a different efficiency that the one of the current year)
            are added to each yearly database.
        """
        if 'model' in time_steps[0] and 'lifetime' in time_steps[0]:
            super().__init__(
                model=time_steps[0]['model'],
                lifetime=time_steps[0]['lifetime'],
                main_database=time_steps[0]['main_database'],
                *args,
                **kwargs,
            )
        elif 'lifetime' in time_steps[0]:
            super().__init__(
                lifetime=time_steps[0]['lifetime'],
                main_database=time_steps[0]['main_database'],
                *args,
                **kwargs,
            )
        elif 'model' in time_steps[0]:
            super().__init__(
                model=time_steps[0]['model'],
                main_database=time_steps[0]['main_database'],
                *args,
                **kwargs,
            )
        else:
            super().__init__(
                main_database=time_steps[0]['main_database'],
                *args,
                **kwargs,
            )

        self.time_steps = time_steps
        self.pathway = True
        self.year = None
        self.list_of_years = [time_step['year'] for time_step in self.time_steps]
        self.operation_metrics_for_all_time_steps = operation_metrics_for_all_time_steps

        self.time_steps = sorted(self.time_steps, key=lambda x: x['year'])  # Sort time steps by year

        list_mapping_time_steps = []

        mapping_copy = self.mapping.copy()
        mapping_copy['Year'] = self.time_steps[0]['year']
        list_mapping_time_steps.append(mapping_copy)

        for i in range(1, len(self.time_steps)):  # Iterate over all time steps but the first one

            self.mapping['Database'] = self.mapping['Database'].replace(
                self.time_steps[i-1]['main_database'].db_names,
                self.time_steps[i]['main_database'].db_names,
            )

            mapping_copy = self.mapping.copy()
            mapping_copy['Year'] = self.time_steps[i]['year']
            list_mapping_time_steps.append(mapping_copy)  # Store the mapping with new codes for each time step

        self.mapping = pd.concat(list_mapping_time_steps, ignore_index=True)  # Concatenate all mappings
        self.mapping.drop_duplicates(inplace=True)  # Remove duplicates for the current time step

    def change_location_mapping_file(self) -> None:

        list_mapping_time_steps = []
        mapping_all_time_steps = self.mapping.copy()

        for i in range(len(self.time_steps)):  # Iterate over all time steps
            time_step = self.time_steps[i]
            year = time_step['year']
            self.mapping = mapping_all_time_steps[mapping_all_time_steps['Year'] == year].copy()
            self.main_database = time_step['main_database']

            super().change_location_mapping_file()

            mapping_copy = self.mapping.copy()
            list_mapping_time_steps.append(mapping_copy)  # Store the mapping with new codes for each time step

        self.mapping = pd.concat(list_mapping_time_steps, ignore_index=True)  # Concatenate all mappings


    def create_esm_database(
            self,
            return_database: bool = False,
            write_database: bool = True,
            *args, **kwargs
    ) -> Database | None:

        all_esm_databases = Database(db_as_list=[])

        # Store the original ESM variable values
        original_esm_db_name = self.esm_db_name
        original_results_path_file = self.results_path_file
        mapping_all_time_steps = self.mapping.copy()

        self.year = self.time_steps[0]['year']
        self.esm_db_name += f'_{self.year}'
        self.results_path_file += f'{self.year}/'

        if self.operation_metrics_for_all_time_steps and len(self.time_steps) == 1:
            raise ValueError("You must have at least two time steps to set 'operation_metrics_for_all_time_steps' to True.")

        for i in range(len(self.time_steps)):  # Iterate over all time steps

            time_step = self.time_steps[i]

            # Update the ESM variable values for the current time step
            self.esm_db_name = self.esm_db_name.replace(str(self.year), str(time_step['year']))
            self.results_path_file = self.results_path_file.replace(str(self.year), str(time_step['year']))

            self.year = time_step['year']
            if 'model' in time_step:
                self.model = time_step['model']
            self.main_database = time_step['main_database']
            if 'main_database_name' in time_step:
                self.main_database_name = time_step['main_database_name']
            else:
                self.main_database_name = self.main_database.db_names
            self.mapping = mapping_all_time_steps[mapping_all_time_steps['Year'] == self.year].copy()

            # create the ESM database for the current time step
            if self.operation_metrics_for_all_time_steps:
                esm_db = super().create_esm_database(
                    return_database=True,
                    write_database=False,
                    *args, **kwargs
                )
                all_esm_databases += esm_db  # concatenate all ESM databases created for each time step

            elif return_database:
                esm_db = super().create_esm_database(
                    return_database=return_database,
                    write_database=write_database,
                    *args, **kwargs
                )
                all_esm_databases += esm_db  # concatenate all ESM databases created for each time step

            else:
                super().create_esm_database(
                    return_database=return_database,
                    write_database=write_database,
                    *args, **kwargs
                )

        # Restore the original ESM variable values
        self.esm_db_name = original_esm_db_name
        self.results_path_file = original_results_path_file
        self.mapping = mapping_all_time_steps

        # add operation metrics for all time steps if requested
        if self.operation_metrics_for_all_time_steps:
            all_esm_databases = self._add_operation_metrics_for_previous_time_steps(
                all_esm_databases=all_esm_databases,
                write_database=write_database,
            )

        if return_database:
            # returns the concatenation of all ESM databases created for each time step
            return all_esm_databases

    def _add_operation_metrics_for_previous_time_steps(
            self,
            all_esm_databases: Database,
            write_database: bool,
    ) -> Database:

        # Store the original ESM variable values
        original_esm_db_name = self.esm_db_name

        year = self.time_steps[0]['year']
        self.esm_db_name += f'_{year}'

        if write_database:
            # Load the completed ESM database for the current year
            esm_db_current_year = Database(
                db_as_list=[i for i in all_esm_databases.db_as_list if i['database'] == self.esm_db_name])

            # Write the ESM database for the current year to Brightway
            esm_db_current_year.write_to_brightway(self.esm_db_name)

        for i in range(1, len(self.time_steps)):  # Iterate over all time steps but the first one
            current_year = self.time_steps[i]['year']
            previous_year = self.time_steps[i-1]['year']
            main_database_current_year = self.time_steps[i]['main_database']
            main_database_name_current_year = main_database_current_year.db_names
            main_database_previous_year = self.time_steps[i-1]['main_database']
            main_database_name_previous_year = main_database_previous_year.db_names

            # Load the ESM database for the previous year (operation datasets only)
            esm_db_previous_year = Database(db_as_list=[
                i for i in copy.deepcopy(all_esm_databases.db_as_list) if
                (i['database'] == self.esm_db_name)
                & (', Construction' not in i['name'])  # Exclude construction activities
                & (', Resource' not in i['name'])  # Exclude resource activities
            ])

            # Rename datasets in the previous year ESM database
            for act in esm_db_previous_year.db_as_list:
                if act['name'].endswith(', Operation'):
                    act['name'] = act['name'].replace(', Operation', f', Operation ({previous_year})')

            # Update the ESM variable values for the current time step
            self.esm_db_name = self.esm_db_name.replace(str(previous_year), str(current_year))

            esm_db_previous_year.relink(
                name_database_unlink=main_database_name_previous_year,
                name_database_relink=main_database_name_current_year,
                database_relink_as_list=main_database_current_year.db_as_list,
                based_on='name',
            )

            # Change database name in the previous year ESM database
            for act in esm_db_previous_year.db_as_list:
                act['database'] = self.esm_db_name
                for exc in act['exchanges']:
                    if (
                            (', Construction' not in exc['name'])
                            & (exc['amount'] != 0)
                            & (exc['database'] == self.esm_db_name.replace(str(current_year), str(previous_year)))
                    ):
                        exc['database'] = self.esm_db_name
                        if 'input' in exc.keys():
                            exc['input'] = (self.esm_db_name, exc['input'][1])

            # Add the relinked previous year ESM database (operation datasets only) to the current year ESM database
            all_esm_databases += esm_db_previous_year

            # Load the completed ESM database for the current year
            esm_db_current_year = Database(db_as_list=[i for i in all_esm_databases.db_as_list if i['database'] == self.esm_db_name])

            if write_database:
                # Write the ESM database for the current year to Brightway
                esm_db_current_year.write_to_brightway(self.esm_db_name)

        # Restore the original ESM variable values
        self.esm_db_name = original_esm_db_name

        return all_esm_databases

    def compute_impact_scores(
            self,
            esm_db_name: str = None,
            *args, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:

        list_impact_scores_time_steps = []
        list_contrib_analysis_time_steps = []

        # Store the original ESM variable values
        original_esm_db_name = self.esm_db_name
        mapping_all_time_steps = self.mapping.copy()
        original_results_path_file = self.results_path_file

        self.year = self.time_steps[0]['year']
        if esm_db_name is not None:
            self.esm_db_name = esm_db_name
        else:
            self.esm_db_name += f'_{self.year}'
        self.results_path_file += f'{self.year}/'

        for i in range(len(self.time_steps)):

            time_step = self.time_steps[i]

            # Update the ESM variable values for the current time step
            self.esm_db_name = self.esm_db_name.replace(str(self.year), str(time_step['year']))
            self.esm_db = Database(db_names=self.esm_db_name)
            if 'lifetime' in time_step:
                self.lifetime = time_step['lifetime']
            self.results_path_file = self.results_path_file.replace(str(self.year), str(time_step['year']))
            self.df_activities_subject_to_double_counting = pd.read_csv(f"{self.results_path_file}activities_subject_to_double_counting.csv")

            self.year = time_step['year']
            self.main_database = time_step['main_database']
            if 'main_database_name' in time_step:
                self.main_database_name = time_step['main_database_name']
            else:
                self.main_database_name = self.main_database.db_names

            if self.operation_metrics_for_all_time_steps:
                self.mapping = mapping_all_time_steps[
                    (mapping_all_time_steps['Year'] == self.year)
                    | ((mapping_all_time_steps['Year'] < self.year) & (mapping_all_time_steps['Type'] == 'Operation'))
                ].copy()
            else:
                self.mapping = mapping_all_time_steps[mapping_all_time_steps['Year'] == self.year].copy()

            # Compute impact scores for the current time step
            impact_scores, contrib_analysis = super().compute_impact_scores(*args, **kwargs)
            impact_scores['Year'] = self.year

            if self.operation_metrics_for_all_time_steps:
                impact_scores = impact_scores.merge(self.mapping[['New_code', 'Year']], on='New_code', suffixes=('', '_inst'))
                # impact_scores['Name'] = impact_scores.apply(
                #     lambda x: f'{x["Name"]} ({x["Year_inst"]})' if x["Year_inst"] < x["Year"] else x["Name"], axis=1)

            list_impact_scores_time_steps.append(impact_scores)

            if contrib_analysis is not None:
                contrib_analysis['Year'] = self.year

                if self.operation_metrics_for_all_time_steps:
                    contrib_analysis = contrib_analysis.merge(
                        self.mapping[['New_code', 'Year']],
                        left_on='act_code',
                        right_on='New_code',
                        suffixes=('', '_inst')
                    )
                    # contrib_analysis['act_name'] = contrib_analysis.apply(
                    #     lambda x: f'{x["act_name"]} ({x["Year_inst"]})' if x["Year_inst"] < x["Year"] else x["act_name"],
                    #     axis=1)

                list_contrib_analysis_time_steps.append(contrib_analysis)

        impact_scores = pd.concat(list_impact_scores_time_steps, ignore_index=True)

        if len(list_contrib_analysis_time_steps) > 0:
            contrib_analysis = pd.concat(list_contrib_analysis_time_steps, ignore_index=True)
        else:
            contrib_analysis = None

        # Restore the original ESM variable values
        self.mapping = mapping_all_time_steps
        self.esm_db_name = original_esm_db_name
        self.results_path_file = original_results_path_file

        return impact_scores, contrib_analysis

    def create_new_database_with_esm_results(
            self,
            esm_results: pd.DataFrame,
            esm_results_db_name: str = None,
            return_database: bool = False,
            *args, **kwargs
    ) -> Database | None:

        all_esm_results_databases = Database(db_as_list=[])

        # Store the original ESM variable values
        original_esm_db_name = self.esm_db_name
        if esm_results_db_name is not None:
            self.esm_results_db_name = esm_results_db_name
        original_esm_results_db_name = self.esm_results_db_name
        mapping_all_time_steps = self.mapping.copy()
        original_results_path_file = self.results_path_file

        self.year = self.time_steps[0]['year']
        self.esm_db_name += f'_{self.year}'
        self.esm_results_db_name += f'_{self.year}'
        self.results_path_file += f'{self.year}/'

        self.main_database = Database(db_as_list=[])  # Initialize main_database to empty Database
        self.esm_db = Database(db_as_list=[])  # Initialize esm_db to empty Database
        self.df_flows_set_to_zero = pd.DataFrame()  # Initialize df_flows_set_to_zero to empty DataFrame

        for i in range(len(self.time_steps)):
            time_step = self.time_steps[i]

            # Update the ESM variable values for the current time step
            if 'lifetime' in time_step:
                self.lifetime = time_step['lifetime']
            self.esm_db_name = self.esm_db_name.replace(str(self.year), str(time_step['year']))
            self.esm_results_db_name = self.esm_results_db_name.replace(str(self.year), str(time_step['year']))
            self.results_path_file = self.results_path_file.replace(str(self.year), str(time_step['year']))
            self.double_counting_removal_amount = pd.read_csv(f'{self.results_path_file}double_counting_removal.csv')

            self.year = time_step['year']
            self.model = time_step['model']
            if 'main_database_name' in time_step:
                self.main_database_name = time_step['main_database_name']
            else:
                self.main_database_name = self.main_database.db_names

            df_flows_set_to_zero = pd.read_csv(f'{self.results_path_file}removed_flows_list.csv')
            df_flows_set_to_zero['Year'] = self.year

            if self.operation_metrics_for_all_time_steps:
                self.df_flows_set_to_zero = pd.concat([self.df_flows_set_to_zero, df_flows_set_to_zero],
                                                      ignore_index=True)
                self.main_database += time_step['main_database']
                self.esm_db += Database(db_names=self.esm_db_name)
                self.mapping = mapping_all_time_steps[
                    (mapping_all_time_steps['Year'] == self.year)
                    | ((mapping_all_time_steps['Year'] < self.year) & (mapping_all_time_steps['Type'] == 'Operation'))
                    ].copy()
            else:
                self.df_flows_set_to_zero = df_flows_set_to_zero
                self.main_database = time_step['main_database']
                self.esm_db = Database(db_names=self.esm_db_name)
                self.mapping = mapping_all_time_steps[mapping_all_time_steps['Year'] == self.year].copy()

            if return_database:
                esm_results_db = super().create_new_database_with_esm_results(
                    return_database=return_database,
                    esm_results=esm_results[esm_results.Year == self.year],
                    *args, **kwargs
                )
                all_esm_results_databases += esm_results_db
            else:
                super().create_new_database_with_esm_results(
                    return_database=return_database,
                    esm_results=esm_results[esm_results.Year == self.year],
                    *args, **kwargs
                )

        # Restore the original ESM variable values
        self.mapping = mapping_all_time_steps
        self.esm_db_name = original_esm_db_name
        self.esm_results_db_name = original_esm_results_db_name
        self.results_path_file = original_results_path_file

        if return_database:
            return all_esm_results_databases

    def connect_esm_results_to_database(
            self,
            esm_results_db_name: str = None,
            specific_db_name: str = None,
            *args, **kwargs
    ) -> None:

        # Store the original ESM variable values
        original_esm_results_db_name = self.esm_results_db_name
        mapping_all_time_steps = self.mapping.copy()

        year = self.time_steps[0]['year']
        if esm_results_db_name is not None:
            self.esm_results_db_name = esm_results_db_name
        else:
            self.esm_results_db_name += f'_{year}'

        if specific_db_name is not None:
            super().connect_esm_results_to_database(specific_db_name=specific_db_name, *args, **kwargs)

        else:
            for i in range(len(self.time_steps)-1):  # Iterate over all time steps except the last one
                current_time_step = self.time_steps[i]
                next_time_step = self.time_steps[i+1]

                # Update the ESM variable values for the current time step
                self.esm_results_db_name = self.esm_results_db_name.replace(str(year), str(current_time_step['year']))

                # Results of time step i are injected in the database of time step i + 1
                year = current_time_step['year']
                self.main_database = next_time_step['main_database']
                self.model = current_time_step['model']
                if 'main_database_name' in next_time_step:
                    self.main_database_name = next_time_step['main_database_name']
                else:
                    self.main_database_name = self.main_database.db_names
                self.mapping = mapping_all_time_steps[mapping_all_time_steps['Year'] == year].copy()

                super().connect_esm_results_to_database(*args, **kwargs)

        # Restore the original ESM variable values
        self.mapping = mapping_all_time_steps
        self.esm_results_db_name = original_esm_results_db_name