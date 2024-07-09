import bw2data as bd


def change_carbon_flow(db_name: str, activity_name: str) -> None:
    """
    Change the biogenic carbon flow of the DAC technology to a fossil carbon flow

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed
    :return: None
    """
    act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
        (activity_name == i.as_dict()['name'])
    )][0]

    biosphere_flows = [i for i in act.biosphere()]

    if len(biosphere_flows) == 2:  # capture and leak:
        if biosphere_flows[0].as_dict()['amount'] == 1:
            uptake_flow = biosphere_flows[0]
            leak_flow = biosphere_flows[1]
        elif biosphere_flows[1].as_dict()['amount'] == 1:
            uptake_flow = biosphere_flows[1]
            leak_flow = biosphere_flows[0]
        else:
            raise ValueError("No flow with amount 1 found")

        # carbon captured (negative emission)
        uptake_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        uptake_flow.as_dict()['categories'] = ('air',)
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0
        uptake_flow.save()

        # carbon leak
        leak_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        leak_flow.as_dict()['categories'] = ('air',)
        leak_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        leak_flow.save()

        act.save()

    else:  # only capture
        uptake_flow = biosphere_flows[0]
        uptake_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        uptake_flow.as_dict()['categories'] = ('air',)
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0
        uptake_flow.save()
        act.save()
