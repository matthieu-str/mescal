from .location_selection import *
from .utils import *


def regionalize_activity_foreground(act: dict, mismatch_regions: list[str], target_region: str,
                                    locations_ranking: list[str], db: list[dict], db_dict_code: dict,
                                    db_dict_name: dict) -> dict:
    """
    Regionalize a foreground activity according to the user ranking of locations

    :param act: activity to regionalize
    :param mismatch_regions: list of regions that are not satisfying the user
    :param target_region: region to which the activity should be regionalized
    :param locations_ranking: list of preferred locations
    :param db: list of activities in the LCI database
    :param db_dict_code: dictionary of the LCI database with (database, code) as key
    :param db_dict_name: dictionary of the LCI database with (name, product, location, database) as key
    :return: the regionalized activity
    """
    if act['location'] in mismatch_regions:  # if we are not satisfied with the current location

        act['comment'] = f'This LCI dataset has been adapted to {target_region}. ' + act.get('comment', '')
        act['location'] = target_region
        prod_flow = get_production_flow(act)
        prod_flow['location'] = target_region

        technosphere_flows = get_technosphere_flows(act)

        # for each technosphere flow, we choose the best possible location according to the user ranking
        for flow in technosphere_flows:
            techno_act = db_dict_code[(flow['database'], flow['code'])]
            techno_act_name = techno_act['name']
            techno_act_product = techno_act['reference product']
            techno_act_database = techno_act['database']

            new_location = change_location_activity(
                esm_tech_name=None,
                product=techno_act_product,
                activity=techno_act_name,
                database=techno_act_database,
                locations_ranking=locations_ranking,
                db=db,
                esm_region=target_region
            )  # best possible location according to the user ranking

            new_techno_act = db_dict_name[(techno_act_name, techno_act_product, new_location, techno_act_database)]
            flow['database'] = new_techno_act['database']
            flow['code'] = new_techno_act['code']
            flow['location'] = new_techno_act['location']

    else:  # nothing changes
        pass

    return act
