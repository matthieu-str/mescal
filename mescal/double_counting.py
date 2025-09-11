import copy
import ast
from .database import Dataset
from .utils import random_code
import pandas as pd
from tqdm import tqdm


def _background_search(
        self,
        act: dict,
        k: int,
        k_lim: int,
        amount: float,
        explore_type: str,
        ESM_inputs: list[str] or str,
        perform_d_c: list[list],
        db_dict_code: dict,
        db_dict_name: dict,
        db_as_list: list[dict],
        db_type: str = 'esm',
) -> tuple[list[list], dict, dict, list[dict]]:
    """
    Explores the tree of the market activity with a recursive approach and write the activities to actually check for
    double-counting in the list perform_d_c.

    :param act: LCI dataset
    :param k: tree depth of act with respect to starting activity
    :param k_lim: maximum allowed tree depth (i.e., maximum recursion depth)
    :param amount: product of amounts when going down in the tree
    :param explore_type: can be 'market' or 'background_removal'
    :param ESM_inputs: list of the ESM flows to perform double counting removal on
    :param perform_d_c: list of activities to check for double counting
    :param db_dict_code: LCI database as dictionary of activities with (database, code) as key
    :param db_dict_name: LCI database as dictionary of activities with (name, reference product, location, database) as key
    :param db_as_list: LCI database as list of activities
    :param db_type: type of database to use, either 'esm', 'esm results' or 'esm results wo dcr'
    :return: list of activities to check for double counting,
        dictionary of activities with code as key,
        dictionary of activities with name as key,
        list of activities
    """

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    if db_type == 'esm':
        esm_db_name = self.esm_db_name
    elif db_type in ['esm results', 'esm results wo dcr']:
        esm_db_name = self.esm_results_db_name
    else:
        raise ValueError('db_type should be either "esm", "esm results", or "esm results wo dcr"')

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
        self.products_without_a_cpc_category.add(act["reference product"])
        self.logger.warning(f'Product {act["reference product"]} has no CPC category.')
        CPC_cat = 'None'

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
        if 'market for' in act['name'] or 'market group for' in act['name']:
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
                                # Modify and save the activity in the ESM database
                                new_act = copy.deepcopy(techno_act)
                                new_code = random_code()
                                new_act['database'] = esm_db_name
                                new_act['code'] = new_code
                                prod_flow = Dataset(new_act).get_production_flow()
                                prod_flow['code'] = new_code
                                prod_flow['database'] = esm_db_name
                                db_as_list.append(new_act)
                                db_dict_name[(
                                    new_act['name'],
                                    new_act['reference product'],
                                    new_act['location'],
                                    new_act['database']
                                )] = new_act
                                db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                # Modify the flow between the activity and its inventory
                                flow['database'] = esm_db_name
                                flow['code'] = new_code
                                flow['input'] = (esm_db_name, new_code)

                                if k < k_lim:  # we continue until maximum depth is reached:
                                    perform_d_c, db_dict_code, db_dict_name, db_as_list = self._background_search(
                                        act=new_act,
                                        k=k + 1,
                                        k_lim=k_lim,
                                        amount=amount * flow['amount'],
                                        explore_type='market',
                                        ESM_inputs=ESM_inputs,
                                        perform_d_c=perform_d_c,
                                        db_dict_code=db_dict_code,
                                        db_dict_name=db_dict_name,
                                        db_as_list=db_as_list,
                                        db_type=db_type,
                                    )
                                    # adding 1 to the current depth k and multiply amount by the flow's amount
                                else:
                                    # if the limit is reached, we consider the last activity for double counting removal
                                    if db_type != 'esm results wo dcr':
                                        new_act['comment'] = (f"Subject to double-counting removal. "
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
                            & (flow['product'] not in ['tap water', 'water, deionised'])
                    ):
                        # we do not consider construction, transport, water and energy flows (we typically target fuel
                        # flows in kg or m3) as well as negative flows
                        techno_act = db_dict_code[(flow['database'], flow['code'])]
                        if 'classifications' in techno_act.keys():
                            if 'CPC' in dict(techno_act['classifications']):
                                new_act = copy.deepcopy(techno_act)
                                new_code = random_code()
                                new_act['database'] = esm_db_name
                                new_act['code'] = new_code
                                prod_flow = Dataset(new_act).get_production_flow()
                                prod_flow['code'] = new_code
                                prod_flow['database'] = esm_db_name
                                db_as_list.append(new_act)
                                db_dict_name[(
                                    new_act['name'],
                                    new_act['reference product'],
                                    new_act['location'],
                                    new_act['database']
                                )] = new_act
                                db_dict_code[(new_act['database'], new_act['code'])] = new_act

                                # Modify the flow between the activity and its inventory and save it
                                flow['database'] = esm_db_name
                                flow['code'] = new_code
                                flow['input'] = (esm_db_name, new_code)

                                if k < k_lim:  # we continue until maximum depth is reached:
                                    perform_d_c, db_dict_code, db_dict_name, db_as_list = self._background_search(
                                        act=new_act,
                                        k=k+1,
                                        k_lim=k_lim,
                                        amount=amount*flow['amount'],
                                        explore_type='market',
                                        ESM_inputs=ESM_inputs,
                                        perform_d_c=perform_d_c,
                                        db_dict_code=db_dict_code,
                                        db_dict_name=db_dict_name,
                                        db_as_list=db_as_list,
                                        db_type=db_type,
                                    )
                                    # here we want to check whether the next activity is a market or not, if not,
                                    # the activity will be added for double counting
                                else:
                                    # if the limit is reached, we consider the last activity for double counting removal
                                    if db_type != 'esm results wo dcr':
                                        new_act['comment'] = (f"Subject to double-counting removal. "
                                                              + new_act.get('comment', ''))
                                        perform_d_c.append([new_act['name'], new_act['code'],
                                                            amount * flow['amount'], k + 1, ESM_inputs])
                            else:
                                self.products_without_a_cpc_category.add(techno_act["reference product"])
                                self.logger.warning(f'Product {techno_act["reference product"]} has no CPC category.')
                        else:
                            self.products_without_a_cpc_category.add(techno_act["reference product"])
                            self.logger.warning(f'Product {techno_act["reference product"]} has no CPC category.')
        return perform_d_c, db_dict_code, db_dict_name, db_as_list

    else:  # the activity is not a market, thus it is added to the list for double-counting removal
        if db_type != 'esm results wo dcr':
            act['comment'] = f"Subject to double-counting removal. " + act.get('comment', '')
            perform_d_c.append([act['name'], act['code'], amount, k, ESM_inputs])
        return perform_d_c, db_dict_code, db_dict_name, db_as_list


def _double_counting_removal(
        self,
        df: pd.DataFrame,
        N: int,
        ESM_inputs: list[str] or str = 'all',
        db_type: str = 'esm',
        ds_type: str = 'Operation',
) -> tuple[list[list], dict, list[list]]:
    """
    Remove double counting in the ESM database and write it in the Brightway project

    :param df: mapping file with input flows of each technology or resource
    :param N: number of columns of the original mapping file
    :param ESM_inputs: list of the ESM flows to perform double counting removal on
    :param db_type: type of database to use, either 'esm', 'esm results' or 'esm results wo dcr'
    :param ds_type: type of LCI dataset to consider, can be 'Operation', 'Construction' or 'Resource'
    :return: list of removed flows, dictionary of removed quantities, list of activities subject to double counting
    """
    # Store frequently accessed instance variables in local variables inside a method.
    # Those will be passed to the background_search method via arguments.
    db_as_list = self.main_database.db_as_list
    db_dict_code = self.main_database.db_as_dict_code
    db_dict_name = self.main_database.db_as_dict_name

    # Store frequently accessed instance variables in local variables inside a method if they don't need to be modified
    esm_db_name = self.esm_db_name
    no_construction_list = self.no_construction_list
    mapping_constr = self.mapping_constr
    no_background_search_list = self.no_background_search_list
    no_double_counting_removal_list = self.no_double_counting_removal_list
    regionalize_foregrounds = self.regionalize_foregrounds
    background_search_act = self.background_search_act

    # Initializing list of removed flows
    flows_set_to_zero = []

    # Initializing the list of activities subject to double counting
    activities_subject_to_double_counting = []

    # Initializing the dict of removed quantities
    ei_removal = {}
    for tech in list(df.Name):
        ei_removal[tech] = {}
        for res in list(df.iloc[:, N:].columns):
            ei_removal[tech][res] = {}
            ei_removal[tech][res]['amount'] = {}
            ei_removal[tech][res]['count'] = {}

    # readings lists as lists and not strings
    try:
        self.technology_compositions.Components = self.technology_compositions.Components.apply(ast.literal_eval)
    except ValueError:
        pass

    try:
        self.mapping_esm_flows_to_CPC_cat.CPC = self.mapping_esm_flows_to_CPC_cat.CPC.apply(ast.literal_eval)
    except ValueError:
        pass

    technology_compositions_dict = {key: value for key, value in dict(zip(
            self.technology_compositions.Name, self.technology_compositions.Components
        )).items()}

    # inverse mapping dictionary (i.e., from CPC categories to the ESM flows)
    mapping_esm_flows_to_CPC_dict = {key: value for key, value in dict(zip(
        self.mapping_esm_flows_to_CPC_cat.Flow, self.mapping_esm_flows_to_CPC_cat.CPC
    )).items()}
    mapping_CPC_to_esm_flows_dict = {}
    for k, v in mapping_esm_flows_to_CPC_dict.items():
        for x in v:
            mapping_CPC_to_esm_flows_dict.setdefault(x, []).append(k)

    for i in tqdm(range(len(df))):
        tech = df['Name'].iloc[i]  # name of ES technology
        # print(tech)

        # Initialization of the list of construction activities and corresponding CPC categories
        act_constr_list = []
        CPC_constr_list = []
        mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] = []

        # Construction activity
        if tech in no_construction_list or ds_type != 'Operation':
            pass

        else:
            if tech not in technology_compositions_dict.keys():  # if the technology is not a composition
                # simple technologies are seen as compositions of one technology
                technology_compositions_dict[tech] = [tech]

            for sub_comp in technology_compositions_dict[tech]:  # looping over the subcomponents of the composition
                database_constr = mapping_constr[mapping_constr.Name == sub_comp]['Database'].iloc[0]
                current_code_constr = mapping_constr[mapping_constr.Name == sub_comp]['Current_code'].iloc[0]

                act_constr = db_dict_code[(database_constr, current_code_constr)]
                act_constr_list.append(act_constr)
                try:
                    CPC_constr = dict(act_constr['classifications'])['CPC']
                except KeyError:
                    self.products_without_a_cpc_category.add(act_constr["reference product"])
                    self.logger.warning(f'Product {act_constr["reference product"]} has no CPC category.')
                    CPC_constr = 'None'
                CPC_constr_list.append(CPC_constr)
                mapping_esm_flows_to_CPC_dict['OWN_CONSTRUCTION'] += [CPC_constr]
                mapping_CPC_to_esm_flows_dict[CPC_constr] = ['OWN_CONSTRUCTION']

        # Main activity
        database_main = df['Database'].iloc[i]  # LCA database of the technology
        current_code_main = df['Current_code'].iloc[i]  # code in ecoinvent

        # identification of the activity in ecoinvent database
        act = db_dict_code[(database_main, current_code_main)]

        if db_type == 'esm':
            # Copy the activity and change the database (no new activity in original ecoinvent database)
            new_code = df['New_code'].iloc[i]  # new code defined previously
            new_act = copy.deepcopy(act)
            new_act['code'] = new_code
            new_act['database'] = esm_db_name
            prod_flow = Dataset(new_act).get_production_flow()
            prod_flow['code'] = new_code
            prod_flow['database'] = esm_db_name
            db_as_list.append(new_act)
            db_dict_name[
                (new_act['name'], new_act['reference product'],
                 new_act['location'], new_act['database'])
            ] = new_act
            db_dict_code[(new_act['database'], new_act['code'])] = new_act
        else:
            new_act = act

        if tech in no_double_counting_removal_list[ds_type]:
            perform_d_c = []
        elif tech in no_background_search_list[ds_type]:
            if db_type != 'esm results wo dcr':
                new_act['comment'] = f"Subject to double-counting removal. " + new_act.get('comment', '')
                perform_d_c = [[new_act['name'], new_act['code'], 1, 0, ESM_inputs]]
            else:
                perform_d_c = []
        else:
            perform_d_c, db_dict_code, db_dict_name, db_as_list = self._background_search(
                act=new_act,
                k=0,
                k_lim=self.max_depth_double_counting_search,
                amount=1,
                explore_type='market',
                ESM_inputs=ESM_inputs,
                perform_d_c=[],
                db_dict_code=db_dict_code,
                db_dict_name=db_dict_name,
                db_as_list=db_as_list,
                db_type=db_type,
            )  # list of activities to perform double counting removal on

        if db_type == 'esm':
            new_act['name'] = f'{tech}, {ds_type}'  # saving name after market identification
            prod_flow = Dataset(new_act).get_production_flow()
            prod_flow['name'] = f'{tech}, {ds_type}'

        id_d_c = 0
        while id_d_c < len(perform_d_c):

            new_act_op_d_c_code = perform_d_c[id_d_c][1]  # activity code
            new_act_op_d_c_amount = perform_d_c[id_d_c][2]  # multiplying factor as we went down in the tree
            k_deep = perform_d_c[id_d_c][3]  # depth level in the process tree
            new_act_op_d_c = None

            if (esm_db_name, new_act_op_d_c_code) in db_dict_code:
                new_act_op_d_c = db_dict_code[
                    (esm_db_name, new_act_op_d_c_code)]  # activity in the database
            else:
                db_names_list = list(set([a['database'] for a in db_as_list]))
                for db_name in db_names_list:
                    if (db_name, new_act_op_d_c_code) in db_dict_code:
                        new_act_op_d_c = db_dict_code[(db_name, new_act_op_d_c_code)]

            if new_act_op_d_c is None:
                raise ValueError(f"Activity not found: {new_act_op_d_c_code}")

            if ds_type in regionalize_foregrounds:
                db_as_list.remove(new_act_op_d_c)
                new_act_op_d_c = self._regionalize_activity_foreground(act=new_act_op_d_c)
                db_as_list.append(new_act_op_d_c)
                db_dict_name[
                    (new_act_op_d_c['name'], new_act_op_d_c['reference product'],
                     new_act_op_d_c['location'], new_act_op_d_c['database'])
                ] = new_act_op_d_c

            if perform_d_c[id_d_c][4] == 'all':
                # list of inputs in the ESM (i.e., negative flows in layers_in_out)
                ES_inputs = list(df.iloc[:, N:].iloc[i][df.iloc[:, N:].iloc[i] < 0].index)
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
                act_flow = db_dict_code[(database, code)]

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

            if tech in no_construction_list or ds_type != 'Operation':
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

                if flow['amount'] < 0 and flow['unit'] != 'unit':
                    # we keep waste flows (but not infrastructure decommissioning flows, which should be contained
                    # in the infrastructure LCI dataset)
                    continue

                # Keep track of the amount in the original activity as a comment
                old_amount = flow['amount']
                flow['comment'] = f'Original amount: {old_amount}. ' + flow.get('comment', '')
                database = flow['database']
                code = flow['code']
                act_flow = db_dict_code[(database, code)]
                res_categories = mapping_CPC_to_esm_flows_dict.get(dict(act_flow['classifications']).get('CPC', ''), '')

                flows_set_to_zero.append([
                    tech,
                    ds_type,
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

                if db_type == 'esm':
                    if 'OWN_CONSTRUCTION' in res_categories:
                        # replace construction flow input by the one added before in the ESM database
                        # (for validation purposes)
                        for idx, sub_comp in enumerate(technology_compositions_dict[tech]):
                            if code == act_constr_list[idx]['code']:
                                new_code_constr = mapping_constr[mapping_constr.Name == sub_comp].New_code.iloc[0]
                                flow['database'], flow['code'] = esm_db_name, new_code_constr
                                flow['name'] = f'{sub_comp}, Construction'

                # add the removed amount in the ei_removal dict for post-analysis
                for cat in res_categories:
                    if cat in ES_inputs:
                        # only adds the amount for the relevant category
                        # (e.g., ELECTRICITY_MV among [ELECTRICITY_LV, ELECTRICITY_MV, ELECTRICITY_HV]
                        # which share the same CPCs
                        if flow['unit'] not in ei_removal[tech][cat]['amount'].keys():
                            # old amount (e.g., GWh) multiplied by factor as we went down in the tree
                            ei_removal[tech][cat]['amount'][flow['unit']] = old_amount * new_act_op_d_c_amount
                            ei_removal[tech][cat]['count'][flow['unit']] = 1  # count (i.e., number of flows put to zero)
                        else:
                            # old amount (e.g., GWh) multiplied by factor as we went down in the tree
                            ei_removal[tech][cat]['amount'][flow['unit']] += old_amount * new_act_op_d_c_amount
                            ei_removal[tech][cat]['count'][flow['unit']] += 1  # count (i.e., number of flows put to zero)

                # Setting the amount to zero
                flow['amount'] = 0

            # Go deeper in the process tree if some flows were not found.
            # This is not applied to construction datasets, which should be found the foreground inventory.
            missing_ES_inputs = []
            for cat in ES_inputs:
                if (
                        (tech in list(background_search_act[ds_type].keys()))
                        & (cat not in ['CONSTRUCTION', 'OWN_CONSTRUCTION', 'DECOMMISSIONING'])
                        # The two following conditions mean that the background search would stop when some
                        # intermediary flows have already been found for a given esm flow, but some other
                        # similar and relevant flows, further in the process tree, might also be there.
                        & ((sum(ei_removal[tech][cat]['amount'].values()) == 0) | (not self.stop_background_search_when_first_flow_found))
                        & ((sum(ei_removal[tech][cat]['count'].values()) == 0) | (not self.stop_background_search_when_first_flow_found))
                ):
                    missing_ES_inputs.append(cat)

            if len(missing_ES_inputs) > 0:
                if k_deep <= background_search_act[ds_type][tech]:
                    perform_d_c, db_dict_code, db_dict_name, db_as_list = self._background_search(
                        act=new_act_op_d_c,
                        k=k_deep,
                        k_lim=background_search_act[ds_type][tech] - 1,
                        amount=new_act_op_d_c_amount,
                        explore_type='background_removal',
                        ESM_inputs=missing_ES_inputs,
                        perform_d_c=perform_d_c,
                        db_dict_code=db_dict_code,
                        db_dict_name=db_dict_name,
                        db_as_list=db_as_list,
                        db_type=db_type,
                    )

            id_d_c += 1

        activities_subject_to_double_counting.extend([[tech, ds_type, i[0], i[1], i[2]] for i in perform_d_c])

    # Injecting local variables into the instance variables
    self.main_database.db_as_list = db_as_list

    return flows_set_to_zero, ei_removal, activities_subject_to_double_counting

def _validation_double_counting(
        self,
        return_validation_report: bool = True,
        save_validation_report: bool = False,
) -> None or pd.DataFrame:
    """
    Generate a validation report for the double-counting removal process: comparison of quantities removed in LCI
    datasets vs quantities in ESM flows. LCI datasets quantities are converted in ESM units (both in terms of inputs
    and outputs, i.e., quantities of input fuels per functional unit).

    :param return_validation_report: if True, returns a DataFrame with the validation report
    :param save_validation_report: if True, saves the validation report as a CSV file
    :return: None or DataFrame with the validation report if return_validation_report is True
    """
    validation_d_c = self._correct_esm_and_lca_efficiency_differences(
        write_efficiency_report=False,
        return_efficiency_report=True,
        db_type='validation',
    )

    validation_d_c.Flow = validation_d_c.Flow.replace("['", "")
    validation_d_c.Flow = validation_d_c.Flow.replace("']", "")

    validation_d_c['Input difference (ESM unit)'] = validation_d_c['ESM input quantity (ESM unit)'] - validation_d_c['LCA input quantity (ESM unit) aggregated']
    validation_d_c['Input difference (%)'] = 100 * validation_d_c['Input difference (ESM unit)'] / validation_d_c['LCA input quantity (ESM unit)']

    validation_d_c = validation_d_c[[
        'Name',
        'Flow',
        'LCA input product',
        'ESM input quantity (ESM unit)',
        'LCA input quantity (ESM unit)',
        'LCA input quantity (ESM unit) aggregated',
        'Input difference (ESM unit)',
        'Input difference (%)',
        'LCA input quantity (LCA unit)',
        'ESM input unit',
        'LCA input unit',
        'Input conversion factor',
        'ESM output unit',
        'LCA output unit',
        'Output conversion factor',
    ]]

    if save_validation_report:
        validation_d_c.to_csv(f'{self.results_path_file}validation_double_counting.csv', index=False)

    if return_validation_report:
        return validation_d_c