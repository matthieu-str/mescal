from .location_selection import *
from .utils import *
import copy


def regionalize_activity_foreground(act: dict, accepted_locations: list[str], target_region: str,
                                    locations_ranking: list[str], db: list[dict], db_dict_code: dict,
                                    db_dict_name: dict, spatialized_database: bool = False,
                                    spatialized_biosphere_db: list[dict] = None,
                                    db_dict_name_spa_biosphere: dict = None) -> dict:
    """
    Regionalize a foreground activity according to the user ranking of locations

    :param act: activity to regionalize
    :param accepted_locations: list of locations that are sufficient for the user
    :param target_region: region to which the activity should be regionalized
    :param locations_ranking: list of preferred locations
    :param db: list of activities in the LCI database
    :param db_dict_code: dictionary of the LCI database with (database, code) as key
    :param db_dict_name: dictionary of the LCI database with (name, product, location, database) as key
    :param spatialized_database: if True, the activity belongs to a spatialized database (with spatialized
        elementary flows)
    :param spatialized_biosphere_db: list of flows in the spatialized biosphere database
    :param db_dict_name_spa_biosphere: dictionary of the spatialized biosphere database with
        (name, product, location, database) as key
    :return: the regionalized activity
    """
    if act['location'] in accepted_locations:
        new_act = copy.deepcopy(act)

    else:
        new_act = copy.deepcopy(act)
        new_act_name = new_act['name']
        new_act_product = new_act['reference product']
        new_act['comment'] = f'This LCI dataset has been adapted to {target_region}. ' + new_act.get('comment', '')
        new_act['location'] = target_region
        prod_flow = get_production_flow(new_act)
        prod_flow['location'] = target_region

        technosphere_flows = get_technosphere_flows(new_act)

        # for each technosphere flow, we choose the best possible location according to the user ranking
        for flow in technosphere_flows:
            techno_act = db_dict_code[(flow['database'], flow['code'])]
            techno_act_name = techno_act['name']
            techno_act_product = techno_act['reference product']
            techno_act_location = techno_act['location']
            techno_act_database = techno_act['database']

            # if a flow is the same as the product (e.g., markets) we do not change the location to avoid infinite loops
            if (techno_act_name == new_act_name) & (techno_act_product == new_act_product):
                continue

            if techno_act_location in accepted_locations:
                continue

            new_location = change_location_activity(
                product=techno_act_product,
                activity=techno_act_name,
                location=techno_act_location,
                database=techno_act_database,
                locations_ranking=locations_ranking,
                db=db,
                esm_region=target_region,
                activity_type='technosphere',
            )  # best possible location according to the user ranking

            new_techno_act = db_dict_name[(techno_act_name, techno_act_product, new_location, techno_act_database)]
            flow['database'] = new_techno_act['database']
            flow['code'] = new_techno_act['code']
            flow['location'] = new_techno_act['location']
            flow['input'] = (new_techno_act['database'], new_techno_act['code'])
            flow['comment'] = f'Changed from {techno_act_location} to {new_location}' + flow.get('comment', '')

        if spatialized_database:
            biosphere_flows = get_biosphere_flows(new_act)
            for flow in biosphere_flows:
                if flow['database'] == 'biosphere3_regionalized_flows':  # if the biosphere flow is regionalized
                    current_loc = flow['name'].split(', ')[-1]
                    if current_loc in accepted_locations:
                        continue
                    generic_name = ', '.join(flow['name'].split(', ')[:-1])
                    new_location = change_location_activity(
                        activity=generic_name,
                        categories=flow['categories'],
                        location=current_loc,
                        database=flow['database'],
                        locations_ranking=locations_ranking,
                        db=spatialized_biosphere_db,
                        esm_region=target_region,
                        activity_type='biosphere',
                    )  # best possible location according to the user ranking

                    new_flow_name = f"{generic_name}, {new_location}"
                    new_biosphere_act = db_dict_name_spa_biosphere[(new_flow_name,
                                                                    flow['categories'],
                                                                    flow['database'])]
                    flow['name'] = new_biosphere_act['name']
                    flow['code'] = new_biosphere_act['code']
                    flow['input'] = (new_biosphere_act['database'], new_biosphere_act['code'])
                    flow['comment'] = f'Changed from {current_loc} to {new_location}' + flow.get('comment', '')

    return new_act
