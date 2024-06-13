import copy
import pandas as pd
from .regionalization import *
import ast


def create_new_activity(name, act_type, current_code, new_code, database, db, db_dict_name, db_dict_code, esm_db_name,
                        regionalize_foregrounds=False, mismatch_regions=None, target_region=None,
                        locations_ranking=None):
    """
    Create a new LCI dataset for the esm technology or resource
    :param name: (str) name of the technology or resource in the esm
    :param act_type: (str) can be 'Construction', 'Operation', or 'Resource'
    :param current_code: (str) code of the activity in the original LCI database
    :param new_code: (str) code of the new activity in the new LCI database
    :param database: (str) name of the original LCI database
    :param db: (list of dict) LCI database
    :param db_dict_name: (dict) dictionary original LCI database with (name, product, location, database) as key
    :param db_dict_code: (dict) dictionary original LCI database with (database, code) as key
    :param esm_db_name: (str) name of the new LCI database
    :param regionalize_foregrounds: (bool) if True, regionalize the foreground activities
    :param mismatch_regions: (list of str) list of regions to be changed
    :param target_region: (str) target region
    :param locations_ranking: (list of str) ranking of the locations
    :return: (dict) new LCI dataset for the technology or resource
    """

    ds = db_dict_code[(database, current_code)]

    new_ds = copy.deepcopy(ds)
    new_ds['name'] = f'{name}, {act_type}'
    new_ds['code'] = new_code
    new_ds['database'] = esm_db_name
    prod_flow = get_production_flow(new_ds)
    prod_flow['name'] = f'{name}, {act_type}'
    prod_flow['code'] = new_code
    prod_flow['database'] = esm_db_name

    if regionalize_foregrounds:
        new_ds = regionalize_activity_foreground(
            act=new_ds,
            mismatch_regions=mismatch_regions,
            target_region=target_region,
            locations_ranking=locations_ranking,
            db=db,
            db_dict_code=db_dict_code,
            db_dict_name=db_dict_name
        )

    return new_ds


def add_activities_to_database(mapping, act_type, db, db_dict_name, db_dict_code, esm_db_name):
    mapping_type = mapping[mapping['Type'] == act_type]
    for i in range(len(mapping_type)):
        ds = create_new_activity(
            name=mapping_type['Name'].iloc[i],
            act_type=act_type,
            current_code=mapping_type['Current_code'].iloc[i],
            new_code=mapping_type['New_code'].iloc[i],
            database=mapping_type['Database'].iloc[i],
            db=db,
            db_dict_name=db_dict_name,
            db_dict_code=db_dict_code,
            esm_db_name=esm_db_name
        )
        db.append(ds)
    return db


def background_search(act, k, k_lim, amount, explore_type, ES_inputs, db, db_dict_code, db_dict_name, esm_db_name,
                      perform_d_c):
    """
    Explores the tree of the market activity with a recursive approach and write the activities to actually check for
    double-counting in the list perform_d_c.
    :param act: (dict) activity
    :param k: (int) tree depth of act with respect to starting activity
    :param k_lim: (int) maximum allowed tree depth (i.e., maximum recursion depth)
    :param amount: (float) product of amounts when going down in the tree
    :param explore_type: (str) can be 'market' or 'background_removal'
    :param ES_inputs: (list of str) list of ES flows to perform double counting removal on
    :param db: (list of dict) LCI database
    :param db_dict_code: (dict) dictionary LCI database with (database, code) as key
    :param db_dict_name: (dict) dictionary LCI database with (name, product, location, database) as key
    :param esm_db_name: (str) name of the new LCI database
    :param perform_d_c: (list of list) list of activities to check for double counting
    :return: (list of list) list of activities to check for double counting (updated with new activities to explore)
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

    technosphere_flows = get_technosphere_flows(act)  # technosphere flows of the activity
    technosphere_flows_act = [db_dict_code[(flow['database'], flow['code'])] for flow in
                              technosphere_flows]  # activities corresponding to technosphere flows inputs
    biosphere_flows = get_biosphere_flows(act)  # biosphere flows of the activity

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
        if (len(technosphere_flows) > 1) & (sum(['classifications' in j.keys() for j in technosphere_flows_act]) == len(
                technosphere_flows_act)):  # all technosphere flows have the classification key
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
            if flow['database'] == esm_db_name:
                # this means that the same activity is several times in the tree, thus inducing an infinite loop
                pass
            else:
                if explore_type == 'market':
                    techno_act = db_dict_code[(flow['database'], flow['code'])]
                    if 'classifications' in techno_act.keys():
                        if 'CPC' in dict(techno_act['classifications']):
                            CPC_cat_new = dict(techno_act['classifications'])['CPC']
                            if CPC_cat == CPC_cat_new:
                                # Modify and save the activity in energyscope database
                                new_act = copy.deepcopy(techno_act)
                                new_code = random_code()
                                new_act['database'] = esm_db_name
                                new_act['code'] = new_code
                                prod_flow = get_production_flow(new_act)
                                prod_flow['code'] = new_code
                                prod_flow['database'] = esm_db_name
                                db.append(new_act)
                                db_dict_name[(new_act['name'], new_act['reference product'], new_act['location'],
                                              new_act['database'])] = new_act
                                db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                # Modify the flow between the activity and its inventory and save it
                                flow['database'] = esm_db_name
                                flow['code'] = new_code

                                if k < k_lim:  # we continue until maximum depth is reached:
                                    perform_d_c, db, db_dict_code, db_dict_name = background_search(
                                        new_act,
                                        k + 1,
                                        k_lim,
                                        amount * flow['amount'],
                                        'market',
                                        ES_inputs,
                                        db,
                                        db_dict_code,
                                        db_dict_name,
                                        esm_db_name,
                                        perform_d_c
                                    )
                                    # adding 1 to the current depth k and multiply amount by the flow's amount
                                else:
                                    # if the limit is reached, we consider the last activity for double counting removal
                                    perform_d_c.append(
                                        [new_act['name'], new_act['code'], amount * flow['amount'], k + 1, ES_inputs]
                                    )
                    else:
                        pass
                elif explore_type == 'background_removal':
                    if (
                            (flow['amount'] > 0)
                            & (flow['unit'] not in ['unit', 'megajoule', 'kilowatt hour', 'ton kilometer'])
                    ):
                        # we do not consider construction, transport and energy flows (we typically target fuel flows
                        # in kg or m3) as well as negative flows
                        techno_act = db_dict_code[(flow['database'], flow['code'])]
                        if 'classifications' in techno_act.keys():
                            if 'CPC' in dict(techno_act['classifications']):
                                new_act = copy.deepcopy(techno_act)
                                new_code = random_code()
                                new_act['database'] = esm_db_name
                                new_act['code'] = new_code
                                prod_flow = get_production_flow(new_act)
                                prod_flow['code'] = new_code
                                prod_flow['database'] = esm_db_name
                                db.append(new_act)
                                db_dict_name[(new_act['name'], new_act['reference product'], new_act['location'],
                                              new_act['database'])] = new_act
                                db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                # Modify the flow between the activity and its inventory and save it
                                flow['database'] = esm_db_name
                                flow['code'] = new_code

                                if k < k_lim:  # we continue until maximum depth is reached:
                                    perform_d_c, db, db_dict_code, db_dict_name = background_search(
                                        new_act,
                                        k + 1,
                                        k_lim,
                                        amount * flow['amount'],
                                        'market',
                                        ES_inputs,
                                        db,
                                        db_dict_code,
                                        db_dict_name,
                                        esm_db_name,
                                        perform_d_c
                                    )
                                    # here we want to check whether the next activity is a market or not, if not,
                                    # the activity will be added for double counting
                                else:
                                    # if the limit is reached, we consider the last activity for double counting removal
                                    perform_d_c.append(
                                        [new_act['name'], new_act['code'], amount * flow['amount'], k + 1, ES_inputs])
                            else:
                                raise ValueError(f"No CPC cat: ({techno_act['database']}, {techno_act['code']})")
                        else:
                            raise ValueError(f"No CPC cat: ({techno_act['database']}, {techno_act['code']})")

    else:  # the activity is not a market, thus it is added to the list for double-counting removal
        perform_d_c.append([act['name'], act['code'], amount, k, ES_inputs])
        return perform_d_c, db, db_dict_code, db_dict_name


def double_counting_removal(df_op, df_constr, esm_db_name, mapping_esm_flows_to_CPC, technology_compositions_dict,
                            db, db_dict_code, db_dict_name, N, background_search_act, no_construction_list,
                            regionalize_foregrounds=False, mismatch_regions=None, target_region=None,
                            locations_ranking=None):
    # Initializing list of removed flows
    flows_set_to_zero = []

    # Initializing the dict of removed quantities
    ei_removal = {}
    for tech in list(df_op.Name):
        ei_removal[tech] = {}
        for res in list(df_op.iloc[:, N:].columns):
            ei_removal[tech][res] = {}
            ei_removal[tech][res]['amount'] = 0
            ei_removal[tech][res]['count'] = 0

    # inverse mapping dictionary (i.e., from CPC categories to the ESM flows)
    mapping_esm_flows_to_CPC_dict = {key: ast.literal_eval(value) for key, value in dict(zip(
        mapping_esm_flows_to_CPC.Flow, mapping_esm_flows_to_CPC.CPC
    )).items()}
    mapping_CPC_to_esm_flows_dict = {}
    for k, v in mapping_esm_flows_to_CPC_dict.items():
        for x in v:
            mapping_CPC_to_esm_flows_dict.setdefault(x, []).append(k)

    for i in range(len(df_op)):
        tech = df_op['Name'].iloc[i]  # name of ES technology
        print(tech)

        # Initialization of the list of construction activities and corresponding CPC categories
        act_constr_list = []
        CPC_constr_list = []
        mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] = []

        # Construction activity
        if tech in no_construction_list:
            pass

        else:
            if tech not in technology_compositions_dict.keys():  # if the technology is a composition
                # simple technologies are seen as compositions of one technology
                technology_compositions_dict[tech] = [tech]

            for sub_comp in technology_compositions_dict[tech]:  # looping over the subcomponents of the composition

                database_constr = df_constr[df_constr.Name == sub_comp]['Database'].iloc[0]
                current_code_constr = df_constr[df_constr.Name == sub_comp]['Current_code'].iloc[0]

                act_constr = db_dict_code[(database_constr, current_code_constr)]
                act_constr_list.append(act_constr)
                CPC_constr = dict(act_constr['classifications'])['CPC']
                CPC_constr_list.append(CPC_constr)
                mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] += [CPC_constr]
                mapping_CPC_to_esm_flows_dict[CPC_constr] = ['OWN_CONSTRUCTION']

        # Operation activity
        database_op = df_op['Database'].iloc[i]  # LCA database of the operation technology
        current_code_op = df_op['Current_code'].iloc[i]  # code in ecoinvent
        new_code = df_op['New_code'].iloc[i]  # new code defined previously

        # identification of the activity in ecoinvent database
        act_op = db_dict_code[(database_op, current_code_op)]

        # Copy the activity and change the database (no new activity in original ecoinvent database)
        new_act_op = copy.deepcopy(act_op)
        new_act_op['code'] = new_code
        new_act_op['database'] = esm_db_name
        prod_flow = get_production_flow(new_act_op)
        prod_flow['code'] = new_code
        prod_flow['database'] = esm_db_name
        db.append(new_act_op)
        db_dict_name[(new_act_op['name'],
                      new_act_op['reference product'],
                      new_act_op['location'],
                      new_act_op['database'])] = new_act_op
        db_dict_code[(new_act_op['database'], new_act_op['code'])] = new_act_op

        perform_d_c, db, db_dict_code, db_dict_name = background_search(
            act=new_act_op,
            k=0,
            k_lim=10,
            amount=1,
            explore_type='market',
            ES_inputs='all',
            db=db,
            db_dict_code=db_dict_code,
            db_dict_name=db_dict_name,
            esm_db_name=esm_db_name,
            perform_d_c=[]
        )  # list of activities to perform double counting removal on

        new_act_op['name'] = f'{tech}, Operation'  # saving name after market identification
        prod_flow = get_production_flow(new_act_op)
        prod_flow['name'] = f'{tech}, Operation'

        # M = len(perform_d_c)
        id_d_c = 0
        while id_d_c < len(perform_d_c):
            new_act_op_d_c_name = perform_d_c[id_d_c][0]  # activity name
            new_act_op_d_c_code = perform_d_c[id_d_c][1]  # activity code
            new_act_op_d_c_amount = perform_d_c[id_d_c][2]  # multiplying factor as we went down in the tree
            k_deep = perform_d_c[id_d_c][3]  # depth level in the process tree
            new_act_op_d_c = db_dict_code[(esm_db_name, new_act_op_d_c_code)]  # activity in the database

            if regionalize_foregrounds:
                regionalize_activity_foreground(
                    new_act_op_d_c,
                    mismatch_regions,
                    target_region,
                    locations_ranking,
                    db,
                    db_dict_code,
                    db_dict_name
                )

            if perform_d_c[id_d_c][4] == 'all':
                # list of inputs in Energyscope (i.e., negative flows in layers_in_out)
                ES_inputs = list(df_op.iloc[:, N:].iloc[i][df_op.iloc[:, N:].iloc[i] < 0].index)
            else:
                ES_inputs = perform_d_c[id_d_c][4]

            # CPCs corresponding to the ESM list of inputs
            CPC_inputs = list(mapping_esm_flows_to_CPC_dict[input] for input in ES_inputs)
            CPC_inputs = [item for sublist in CPC_inputs for item in sublist]  # flatten the list of lists

            # Creating the list containing the CPCs of all technosphere flows of the activity
            technosphere_inputs = get_technosphere_flows(new_act_op_d_c)
            technosphere_inputs_CPC = []

            for exc in technosphere_inputs:

                # if (exc['amount'] < 0) & (exc['unit'] == 'unit'):
                #     print('Potential decommission flow:', new_act_op_d_c['name'], exc['name'], exc['amount'])

                database = exc['database']
                code = exc['code']
                act_flow = db_dict_code[(database, code)]

                if 'classifications' in list(act_flow.keys()):
                    if 'CPC' in dict(act_flow['classifications']).keys():
                        technosphere_inputs_CPC.append(dict(act_flow['classifications'])['CPC'])
                    else:
                        technosphere_inputs_CPC.append('None')
                else:
                    technosphere_inputs_CPC.append('None')

            # Finding the indices of technosphere flows that are also in Energyscope's inputs
            # (i.e., flows that we want to put to zero)
            set_CPC_inputs = set(CPC_inputs)
            id_technosphere_inputs_zero = [i for i, e in enumerate(technosphere_inputs_CPC) if e in set_CPC_inputs]

            if tech in no_construction_list:
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
                flow['comment'] = f'Original amount: {old_amount}'
                database = flow['database']
                code = flow['code']
                act_flow = db_dict_code[(database, code)]
                res_categories = mapping_CPC_to_esm_flows_dict.get(dict(act_flow['classifications']).get('CPC', ''), '')

                flows_set_to_zero.append([new_act_op_d_c['reference product'],  # activity in which the flow is removed
                                          new_act_op_d_c['name'],
                                          new_act_op_d_c['location'],
                                          new_act_op_d_c['database'],
                                          old_amount, flow['unit'],  # quantity and unit
                                          act_flow['reference product'],  # removed flow
                                          act_flow['name'],
                                          act_flow['location']
                                          ]
                                         )

                if 'OWN_CONSTRUCTION' in res_categories:
                    # replace construction flow input by the one added before in the ES database
                    # (for validation purposes)
                    for idx, sub_comp in enumerate(technology_compositions_dict[tech]):
                        if code == act_constr_list[idx]['code']:
                            new_code_constr = df_constr[df_constr.Name == sub_comp].New_code.iloc[0]
                            flow['database'], flow['code'] = esm_db_name, new_code_constr

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

            # Go deeper in the process tree if some flows were not found
            missing_ES_inputs = []
            for cat in ES_inputs:
                if ((tech in list(background_search_act.keys()))
                        & (cat not in ['CONSTRUCTION', 'OWN_CONSTRUCTION', 'DECOMMISSIONING'])
                        & (ei_removal[tech][cat]['amount'] == 0) & (ei_removal[tech][cat]['count'] == 0)):
                    missing_ES_inputs.append(cat)

            if len(missing_ES_inputs) > 0:
                if k_deep <= background_search_act[tech]:
                    perform_d_c, db, db_dict_code, db_dict_name = background_search(
                        new_act_op_d_c,
                        k_deep,
                        background_search_act[tech] - 1,
                        new_act_op_d_c_amount,
                        'background_removal',
                        missing_ES_inputs,
                        db,
                        db_dict_code,
                        db_dict_name,
                        esm_db_name,
                        perform_d_c
                    )

            id_d_c += 1

    return db, db_dict_code, db_dict_name, flows_set_to_zero, ei_removal


def has_construction(row, no_construction_list):
    if row.Name in no_construction_list:
        return 0
    else:
        return -1


def has_decommissioning(row, decom_list):
    if row.Name in decom_list:
        return -1
    else:
        return 0


def is_transport(row, mobility_list):
    if len(row[row == 1]) == 0:
        return 0
    elif row[row == 1].index[0] in mobility_list:
        return -1
    else:
        return 0


def is_process_activity(row, process_list):
    if row.Name in process_list:
        return -1
    else:
        return 0


def add_technology_specifics(mapping_op, df_tech_specifics):
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

    # Add a background search depth to technologies that need a background search
    activities_background_search = list(df_tech_specifics[df_tech_specifics.Specifics == 'Background search'].Name)
    background_search_act = {}
    for tech in activities_background_search:
        background_search_act[tech] = df_tech_specifics[df_tech_specifics.Name == tech].Amount.iloc[0]

    return mapping_op, background_search_act, no_construction_list


def create_esm_database(mapping, model, tech_specifics, technology_compositions, mapping_esm_flows_to_CPC_cat,
                        main_database, esm_db_name):
    db_dict_name = database_list_to_dict(main_database, 'name')
    db_dict_code = database_list_to_dict(main_database, 'code')

    technology_compositions_dict = {key: ast.literal_eval(value) for key, value in dict(zip(
        technology_compositions.Name, technology_compositions.Components
    )).items()}

    # Adding current code to the mapping file
    mapping['Current_code'] = mapping.apply(lambda row: get_code(
        db_dict_name=db_dict_name,
        product=row['Product'],
        activity=row['Activity'],
        region=row['Location'],
        database=row['Database']
    ), axis=1)

    # Creating a new code for each activity to be added
    mapping['New_code'] = mapping.apply(lambda row: random_code(), axis=1)

    model = model.pivot(index='Name', columns='Flow', values='Amount').reset_index()
    model.fillna(0, inplace=True)

    N = mapping.shape[1]

    mapping_op = mapping[mapping['Type'] == 'Operation']
    mapping_constr = mapping[mapping['Type'] == 'Construction']

    mapping_op = pd.merge(mapping_op, model, on='Name', how='left')
    mapping_op['CONSTRUCTION'] = mapping_op.shape[0] * [0]
    mapping_op, background_search_act, no_construction_list = add_technology_specifics(mapping_op, tech_specifics)

    # Add construction and resource activities to the database (which do not need double counting removal)
    main_database = add_activities_to_database(mapping, 'Construction', main_database, db_dict_name,
                                               db_dict_code, esm_db_name)
    main_database = add_activities_to_database(mapping, 'Resource', main_database, db_dict_name,
                                               db_dict_code, esm_db_name)

    main_database, db_dict_code, db_dict_name, flows_set_to_zero, ei_removal = double_counting_removal(
        df_op=mapping_op,
        df_constr=mapping_constr,
        esm_db_name=esm_db_name,
        mapping_esm_flows_to_CPC=mapping_esm_flows_to_CPC_cat,
        technology_compositions_dict=technology_compositions_dict,
        db=main_database,
        db_dict_code=db_dict_code,
        db_dict_name=db_dict_code,
        N=N,
        background_search_act=background_search_act,
        no_construction_list=no_construction_list,
        regionalize_foregrounds=False,
        mismatch_regions=None,
        target_region=None,
        locations_ranking=None
    )

    esm_db = [act for act in main_database if act['database'] == esm_db_name]

    write_wurst_database_to_brightway(esm_db, esm_db_name)

    df_flows_set_to_zero = pd.DataFrame(data=flows_set_to_zero,
                                        columns=['Product', 'Activity', 'Location', 'Database', 'Amount', 'Unit',
                                                 'Removed flow product', 'Removed flow activity',
                                                 'Removed flows location'])
    df_flows_set_to_zero.drop_duplicates(inplace=True)

    ei_removal_amount = {}
    ei_removal_count = {}
    for tech in list(mapping_op.Name):
        ei_removal_amount[tech] = {}
        ei_removal_count[tech] = {}
        for res in list(mapping_op.iloc[:, N:].columns):
            ei_removal_amount[tech][res] = ei_removal[tech][res]['amount']
            ei_removal_count[tech][res] = ei_removal[tech][res]['count']

    double_counting_removal_amount = pd.DataFrame.from_dict(ei_removal_amount, orient='index')
    double_counting_removal_count = pd.DataFrame.from_dict(ei_removal_count, orient='index')
    double_counting_removal_amount.to_csv("energyscope_data/results/double_counting_removal.csv")
    double_counting_removal_count.to_csv("energyscope_data/results/double_counting_removal_count.csv")
    df_flows_set_to_zero.to_csv("energyscope_data/results/removed_flows_list.csv", index=False)
