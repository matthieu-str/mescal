import pandas as pd
import ast
import copy
import time
from pathlib import Path
from .modify_inventory import change_dac_biogenic_carbon_flow, change_fossil_carbon_flows_of_biofuels
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
            technology_compositions: pd.DataFrame = None,
            results_path_file: str = 'results/',
            tech_specifics: pd.DataFrame = None,
            regionalize_foregrounds: bool = False,
            accepted_locations: list[str] = None,
            esm_location: str = None,
            locations_ranking: list[str] = None,
            spatialized_database: bool = False,
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
        :param results_path_file: path to your result folder
        :param regionalize_foregrounds: if True, regionalize the ESM database foreground
        :param accepted_locations: list of ecoinvent locations to keep without modification in case of regionalization
        :param esm_location: ecoinvent location corresponding to the geographical scope of the ESM
        :param locations_ranking: ranking of the preferred ecoinvent locations in case of regionalization
        :param spatialized_database: set to True if the main database has spatialized elementary flows
        :param spatialized_biosphere_db: spatialized biosphere database (to provide if spatialized_database is True)
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
        self.esm_db_name = esm_db_name
        self.results_path_file = results_path_file
        self.regionalize_foregrounds = regionalize_foregrounds
        self.accepted_locations = accepted_locations
        self.esm_location = esm_location
        self.locations_ranking = locations_ranking
        self.spatialized_database = spatialized_database
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

    # Import methods from other files
    from .regionalization import (
        regionalize_activity_foreground,
        change_location_activity,
        change_location_mapping_file
    )
    from .double_counting import double_counting_removal, background_search
    from .impact_assessment import compute_impact_scores, get_impact_categories, is_empty
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
        replace_mobility_end_use_type
    )
    from .normalization import normalize_lca_metrics
    from .generate_lcia_obj_ampl import generate_mod_file_ampl

    def create_esm_database(
            self,
            return_database: bool = False,
            write_database: bool = True,
            write_double_counting_removal_reports: bool = True,
    ) -> Database:
        """
        Create the ESM database after double counting removal. Three csv files summarizing the double-counting removal
        process are automatically saved in the results folder: double_counting_removal.csv (amount of removed
        flows), double_counting_removal_count.csv (number of flows set to zero), and removed_flows_list.csv
        (specific activities in which the flows were removed).

        :param return_database: if True, return the ESM database as a mescal.Database object
        :param write_database: if True, write the ESM database to Brightway
        :param write_double_counting_removal_reports: if True, write the double-counting removal reports in the results
            folder
        :return: the ESM database if return_database is True
        """

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
        print(f"### Starting to add construction and resource activities database ###")
        t1_add = time.time()
        self.add_activities_to_database(act_type='Construction')
        self.add_activities_to_database(act_type='Resource')
        t2_add = time.time()
        print(f"### Construction and resource activities added to the database ###")
        print(f"--> Time: {round(t2_add - t1_add, 0)} seconds")

        print(f"### Starting to remove double-counted flows ###")
        t1_dc = time.time()
        flows_set_to_zero, ei_removal = self.double_counting_removal(df_op=self.mapping_op, N=N, ESM_inputs='all')
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

        Path(self.results_path_file).mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist

        if write_double_counting_removal_reports:
            double_counting_removal_amount.to_csv(f"{self.results_path_file}double_counting_removal.csv", index=False)
            double_counting_removal_count.to_csv(f"{self.results_path_file}double_counting_removal_count.csv",
                                                 index=False)
            df_flows_set_to_zero.to_csv(f"{self.results_path_file}removed_flows_list.csv", index=False)

        if write_database:
            print(f"### Starting to write database ###")
            t1_mod_inv = time.time()
            esm_db = Database(
                db_as_list=[act for act in self.main_database.db_as_list if act['database'] == self.esm_db_name])
            esm_db.write_to_brightway(self.esm_db_name)

            # The following modifications are made via bw2data and thus need a written database
            # Change carbon flow of DAC from biogenic to fossil
            dac_technologies = list(self.tech_specifics[self.tech_specifics.Specifics == 'DAC'].Name)
            for tech in dac_technologies:
                if f'{tech}, Operation' in [act['name'] for act in esm_db.db_as_list]:
                    change_dac_biogenic_carbon_flow(db_name=self.esm_db_name, activity_name=f'{tech}, Operation')

            # Change carbon flows of biofuel mobility technologies
            biofuel_mob_tech = self.tech_specifics[self.tech_specifics.Specifics == 'Biofuel'][
                ['Name', 'Amount']].values.tolist()
            for tech, biogenic_ratio in biofuel_mob_tech:
                if f'{tech}, Operation' in [act['name'] for act in esm_db.db_as_list]:
                    change_fossil_carbon_flows_of_biofuels(
                        db_name=self.esm_db_name,
                        activity_name=f'{tech}, Operation',
                        biogenic_ratio=biogenic_ratio
                    )
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
