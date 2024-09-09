import bw2data as bd


def change_dac_biogenic_carbon_flow(db_name: str, activity_name: str) -> None:
    """
    Change the biogenic carbon flow of premise DAC technologies to a fossil carbon flow

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed
    :return: None (changes are saved in the database)
    """
    act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
        (activity_name == i.as_dict()['name'])
    )][0]

    biosphere_flows = [i for i in act.biosphere()]

    if len(biosphere_flows) == 2:  # capture and leak:
        if biosphere_flows[0].as_dict()['amount'] == 1:  # the capture flow has amount 1
            uptake_flow = biosphere_flows[0]
            leak_flow = biosphere_flows[1]
        elif biosphere_flows[1].as_dict()['amount'] == 1:
            uptake_flow = biosphere_flows[1]
            leak_flow = biosphere_flows[0]
        else:
            raise ValueError("No flow with amount 1 found")

        # carbon captured (negative emission)
        old_name_uptake = uptake_flow.as_dict()['name']
        uptake_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        uptake_flow.as_dict()['categories'] = ('air',)
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0  # carbon captured (negative emission)
        uptake_flow.as_dict()['comment'] = (f"Modified from {old_name_uptake} to Carbon dioxide, fossil. Amount "
                                            f"multiplied by -1. ") + uptake_flow.as_dict().get('comment', "")
        uptake_flow.save()

        # carbon leak
        old_name_leak = leak_flow.as_dict()['name']
        leak_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        leak_flow.as_dict()['categories'] = ('air',)
        leak_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        leak_flow.as_dict()['comment'] = (f"Modified from {old_name_leak} to Carbon dioxide, fossil. "
                                          + uptake_flow.as_dict().get('comment', ""))
        leak_flow.save()

        act.save()

    elif len(biosphere_flows) == 1:  # only capture
        uptake_flow = biosphere_flows[0]
        old_name_uptake = uptake_flow.as_dict()['name']
        uptake_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        uptake_flow.as_dict()['categories'] = ('air',)
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0  # carbon captured (negative emission)
        uptake_flow.as_dict()['comment'] = (f"Modified from {old_name_uptake} to Carbon dioxide, fossil. Amount "
                                            f"multiplied by -1. ") + uptake_flow.as_dict().get('comment', "")
        uptake_flow.save()
        act.save()

    else:
        raise ValueError(f"Unexpected number of biosphere flows for {act.as_dict()['name']}. Should be 1 or 2, "
                         f"but is {len(biosphere_flows)}")
