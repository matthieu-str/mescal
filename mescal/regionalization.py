from .location_selection import *
from .utils import *


def regionalize_activity_foreground(act, mismatch_regions, target_region, locations_ranking, db, db_dict_code,
                                    db_dict_name):

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
