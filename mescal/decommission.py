from .utils import random_code
import pandas as pd
import ast


def _add_decommission_datasets(
        self,
        add_decom_ds_to_db: bool = True,
) -> None:
    """
    This method aggregates the EoL flows that have been identified during the double-counting removal step in a
    decommission dataset. This is applied to every technology that has not already a decommission adataset.
    The mapping, unit conversion, and technology compositions files are updated accordingly.

    :param add_decom_ds_to_db: If True, decommission datasets are added to the main database. This argument is True
        when decommission datasets are created during the create_esm_database method. This argument is False there is
        a need to recover decommission datasets codes from the ESM database (outside the create_esm_database method).
    :return: None
    """

    if self.df_flows_set_to_zero is None:
        self.df_flows_set_to_zero = pd.read_csv(f'{self.results_path_file}removed_flows_list.csv')

    # Store frequently accessed instance variables in local variables inside a method
    df = self.df_flows_set_to_zero
    mapping_constr = self.mapping_constr
    mapping_decom = self.mapping_decom
    unit_conversion = self.unit_conversion
    technology_compositions = self.technology_compositions
    esm_db_name = self.esm_db_name
    db_as_list = self.main_database.db_as_list
    db_dict_code = self.main_database.db_as_dict_code

    # readings lists as lists and not strings
    try:
        self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    comp_to_tech = {}
    for _, row in technology_compositions.iterrows():
        if row.Type == 'Construction':
            for comp in row.Components:
                comp_to_tech[comp] = row.Name

    new_mapping_data = []
    new_technology_compositions_dict = {}
    df_removed_decom = df[
        (df.Type == 'Construction')
        & (df['Amount (scaled to the FU)'] < 0)  # waste flows only
    ]

    for tech in df_removed_decom.Name.unique():  # iterate over construction technologies

        if tech in mapping_decom.Name.unique():
            continue  # skip technologies that already have a decommission dataset

        if tech in comp_to_tech:
            parent_tech = comp_to_tech[tech]
            if parent_tech in mapping_decom.Name.unique():
                continue  # skip components of technologies that already have a decommission dataset
            else:
                if parent_tech in new_technology_compositions_dict:
                    new_technology_compositions_dict[parent_tech] += [tech]
                else:
                    new_technology_compositions_dict[parent_tech] = [tech]

        act_constr_code, act_constr_database = mapping_constr[mapping_constr.Name == tech][['Current_code', 'Database']].iloc[0]
        act_constr = db_dict_code[(act_constr_database, act_constr_code)]

        if add_decom_ds_to_db:

            df_removed_decom_tech = df_removed_decom[(df_removed_decom.Name == tech)].reset_index(drop=True)
            new_code = random_code()

            exchanges = [{
                'amount': -1,
                'code': new_code,
                'type': 'production',
                'name': f'{tech}, Decommission',
                'product': f'used {act_constr["reference product"]}',
                'unit': act_constr['unit'],
                'location': act_constr['location'],
                'database': esm_db_name,
            }]

            for i in range(len(df_removed_decom_tech)):
                exchanges.append(
                    {
                        'amount': df_removed_decom_tech['Amount (scaled to the FU)'].iloc[i],
                        'code': df_removed_decom_tech['Removed flow code'].iloc[i],
                        'type': 'technosphere',
                        'name': df_removed_decom_tech['Removed flow activity'].iloc[i],
                        'product': df_removed_decom_tech['Removed flow product'].iloc[i],
                        'unit': df_removed_decom_tech['Unit'].iloc[i],
                        'location': df_removed_decom_tech['Removed flow location'].iloc[i],
                        'database': df_removed_decom_tech['Removed flow database'].iloc[i],
                    }
                )

            new_activity = {
                'database': esm_db_name,
                'name': f'{tech}, Decommission',
                'location': act_constr['location'],
                'unit': act_constr['unit'],
                'reference product': f'used {act_constr["reference product"]}',
                'code': new_code,
                'classifications': [('CPC', '39990: Other wastes n.e.c.')],
                'comment': f'Activity derived from the aggregation of waste flows in ({act_constr["reference product"]} '
                           f'- {act_constr["name"]} - {act_constr["location"]})',
                'parameters': {},
                'exchanges': exchanges,
            }

            db_as_list.append(new_activity)

        else:
            # Recover decommission datasets new codes from ESM database
            new_code = [i['code'] for i in self.esm_db.db_as_list if i['name'] == f'{tech}, Decommission'][0]

        new_mapping_data.append([
            tech,
            'Decommission',
            f'used {act_constr["reference product"]}',
            f'{tech}, Decommission',
            act_constr["location"],
            esm_db_name,
            None,
            new_code,
        ])

    new_mapping = pd.DataFrame(
        data=new_mapping_data,
        columns=['Name', 'Type', 'Product', 'Activity', 'Location', 'Database', 'Current_code', 'New_code']
    )

    new_unit_conversion = unit_conversion[
        (unit_conversion.Name.isin([tech for tech in new_mapping.Name.unique()]))
        & (unit_conversion.Type == 'Construction')
    ]
    new_unit_conversion['Type'] = 'Decommission'
    new_unit_conversion['Value'] *= -1  # waste flows

    new_unit_conversion_comp = unit_conversion[
        (unit_conversion.Name.isin([tech for tech in new_technology_compositions_dict.keys()]))
        & (unit_conversion.Type == 'Construction')
    ]
    new_unit_conversion_comp['Type'] = 'Decommission'
    # no change to 'Value' because the -1 factor is already in the unit conversion factors of subcomponents

    new_technology_compositions = pd.DataFrame({
        "Name": list(new_technology_compositions_dict.keys()),
        "Components": list(new_technology_compositions_dict.values()),
    })
    new_technology_compositions['Type'] = 'Decommission'

    self.added_decom_to_input_data = True

    # Injecting local variables into the instance variables
    self.main_database.db_as_list = db_as_list
    self.mapping = pd.concat([self.mapping, new_mapping], ignore_index=True)
    self.unit_conversion = pd.concat([unit_conversion, new_unit_conversion, new_unit_conversion_comp], ignore_index=True)
    self.technology_compositions = pd.concat([technology_compositions, new_technology_compositions], ignore_index=True)