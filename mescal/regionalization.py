from .location_selection import *
from .utils import *
import copy


def regionalize_activity_foreground(act: dict, accepted_locations: list[str], target_region: str,
                                    locations_ranking: list[str], db: list[dict], db_dict_code: dict,
                                    db_dict_name: dict, regionalized_database: bool = False,
                                    regionalized_biosphere_db: list[dict] = None,
                                    db_dict_name_reg_biosphere: dict = None) -> dict:
    """
    Regionalize a foreground activity according to the user ranking of locations

    :param act: activity to regionalize
    :param accepted_locations: list of locations that are sufficient for the user
    :param target_region: region to which the activity should be regionalized
    :param locations_ranking: list of preferred locations
    :param db: list of activities in the LCI database
    :param db_dict_code: dictionary of the LCI database with (database, code) as key
    :param db_dict_name: dictionary of the LCI database with (name, product, location, database) as key
    :param regionalized_database: if True, the activity belongs to a regionalized database (with regionalized
        elementary flows)
    :param regionalized_biosphere_db: list of flows in the regionalized biosphere database
    :param db_dict_name_reg_biosphere: dictionary of the regionalized biosphere database with
        (name, product, location, database) as key
    :return: the regionalized activity
    """
    if act['location'] in accepted_locations:
        new_act = copy.deepcopy(act)

    else:
        new_act = copy.deepcopy(act)
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

        if regionalized_database:
            biosphere_flows = get_biosphere_flows(new_act)
            print(biosphere_flows)
            for flow in biosphere_flows:
                if flow['database'] == 'biosphere3_regionalized_flows':  # if the biosphere flow is regionalized
                    current_loc = flow['name'].split(', ')[-1]
                    generic_name = ', '.join(flow['name'].split(', ')[:-1])
                    print(flow['categories'], generic_name)
                    new_location = change_location_activity(
                        activity=generic_name,
                        categories=flow['categories'],
                        location=current_loc,
                        database=flow['database'],
                        locations_ranking=locations_ranking,
                        db=regionalized_biosphere_db,
                        esm_region=target_region,
                        activity_type='biosphere',
                    )  # best possible location according to the user ranking

                    new_flow_name = f"{generic_name}, {new_location}"
                    new_biosphere_act = db_dict_name_reg_biosphere[(new_flow_name,
                                                                    flow['categories'],
                                                                    flow['database'])]
                    flow['name'] = new_biosphere_act['name']
                    flow['code'] = new_biosphere_act['code']
                    flow['input'] = (new_biosphere_act['database'], new_biosphere_act['code'])

    return new_act
