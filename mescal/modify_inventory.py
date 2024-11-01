import bw2data as bd


def change_dac_biogenic_carbon_flow(
        db_name: str,
        activity_name: str = None,
        activity_code: str = None
) -> None:
    """
    Change the biogenic carbon flow of premise DAC technologies to a fossil carbon flow

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param activity_code: code of the activity to be changed
    :return: None (changes are saved in the database)
    """
    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

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
        uptake_flow.as_dict()['code'] = '349b29d1-3e58-4c66-98b9-9d1a076efd2e'
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0  # carbon captured (negative emission)
        uptake_flow.as_dict()['comment'] = (f"Modified from {old_name_uptake} to Carbon dioxide, fossil. Amount "
                                            f"multiplied by -1. ") + uptake_flow.as_dict().get('comment', "")
        uptake_flow.save()

        # carbon leak
        old_name_leak = leak_flow.as_dict()['name']
        leak_flow.as_dict()['name'] = 'Carbon dioxide, fossil'
        leak_flow.as_dict()['categories'] = ('air',)
        leak_flow.as_dict()['code'] = '349b29d1-3e58-4c66-98b9-9d1a076efd2e'
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
        uptake_flow.as_dict()['code'] = '349b29d1-3e58-4c66-98b9-9d1a076efd2e'
        uptake_flow.as_dict()['input'] = ('biosphere3', '349b29d1-3e58-4c66-98b9-9d1a076efd2e')
        uptake_flow.as_dict()['amount'] *= -1.0  # carbon captured (negative emission)
        uptake_flow.as_dict()['comment'] = (f"Modified from {old_name_uptake} to Carbon dioxide, fossil. Amount "
                                            f"multiplied by -1. ") + uptake_flow.as_dict().get('comment', "")
        uptake_flow.save()
        act.save()

    else:
        raise ValueError(f"Unexpected number of biosphere flows for {act.as_dict()['name']}. Should be 1 or 2, "
                         f"but is {len(biosphere_flows)}")


def change_fossil_carbon_flows_of_biofuels(
        db_name: str,
        activity_name: str = None,
        activity_code: str = None,
        biogenic_ratio: float = 1
) -> None:
    """

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param activity_code: code of the activity to be changed
    :param biogenic_ratio: fraction of biogenic carbon in the biofuel. Default is 1 (100% biogenic).
    :return: None (changes are saved in the database)
    """
    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

    biosphere_flows = [i for i in act.biosphere()]

    for bf in biosphere_flows:
        if bf.as_dict()['name'] in ['Carbon dioxide, fossil', 'Carbon monoxide, fossil', 'Methane, fossil']:

            new_bf_name = bf.as_dict()['name'].replace('fossil', 'non-fossil')

            new_biosphere_act = [bio_act for bio_act in bd.Database('biosphere3').search(new_bf_name, limit=1000) if (
                    (bio_act['name'] == new_bf_name)
                    & (bio_act['categories'] == bf.as_dict()['categories'])
            )][0]  # looking for the equivalent non-fossil flow in biosphere3

            # change the amount of the fossil flow
            total_amount = bf.as_dict()['amount']
            bf.as_dict()['amount'] *= (1 - biogenic_ratio)
            bf['comment'] = (f"Multiplied by (1 - biogenic ratio): {round(1-biogenic_ratio, 3)}. "
                             + bf.get('comment', ""))
            bf.save()

            # add a new non-fossil elementary flow to the activity
            new_biosphere_exc = act.new_exchange(
                input=new_biosphere_act,
                name=new_biosphere_act['name'],
                categories=new_biosphere_act['categories'],
                amount=total_amount * biogenic_ratio,
                type='biosphere',
            )
            new_biosphere_exc['comment'] = (f"Added flow: biogenic part of {bf.as_dict()['name']} with biogenic ratio "
                                            f"of {round(biogenic_ratio, 3)}. ") + new_biosphere_exc.get('comment', "")
            new_biosphere_exc.save()

    act.as_dict()['comment'] = (f"Modified fossil carbon flows to non-fossil carbon flows. Biogenic ratio: "
                                f"{round(biogenic_ratio, 3)}. ") + act.get('comment', "")
    act.save()


def remove_quebec_flow_in_global_heat_market(
        db_name: str,
        activity_name: str = None,
        activity_code: str = None
) -> None:
    """
    Remove the Quebec heat flow in the global heat market activity

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param activity_code: code of the activity to be changed
    :return: None (changes are saved in the database)
    """
    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
            & (i.as_dict()['location'] == 'GLO')
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

    # Delete the exchange and save the amount
    amount = 0
    for exc in act.technosphere():
        if exc['location'] == 'CA-QC':
            amount = exc['amount']
            exc.delete()

    if amount == 0:
        raise ValueError("No exchange with location 'CA-QC' found")

    # Add the deleted amount in the RoW exchange
    for exc in act.technosphere():
        if exc['location'] == 'RoW':
            exc['amount'] += amount
            exc['comment'] = (f"Added the amount of the Quebec exchange to the RoW exchange ({amount}). "
                              + exc.get('comment', ""))
            exc.save()

    act.save()


def change_direct_carbon_emissions_by_factor(
        db_name: str,
        activity_name: str = None,
        activity_code: str = None,
        factor: float = 1
) -> None:
    """
    Change the direct emissions of an activity by a factor

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param activity_code: code of the activity to be changed
    :param factor: factor by which the direct emissions are multiplied
    :return: None (changes are saved in the database)
    """
    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

    for exc in act.biosphere():
        if exc['name'] in [
            'Carbon dioxide, fossil',
            'Carbon monoxide, fossil',
            'Methane, fossil',
            'Carbon dioxide, non-fossil',
            'Carbon monoxide, non-fossil',
            'Methane, non-fossil'
        ]:
            exc['amount'] *= factor
            exc['comment'] = f"Multiplied carbon flows by factor: {round(factor, 3)}. " + exc.get('comment', "")
            exc.save()

    act.save()


def add_carbon_dioxide_flow(
        db_name: str,
        amount: float,
        activity_name: str = None,
        activity_code: str = None,
) -> None:
    """
    Add a carbon dioxide flow to an activity

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param activity_code: code of the activity to be changed
    :param amount: amount of the carbon dioxide flow
    :return: None (changes are saved in the database)
    """
    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

    co2_flow = [bio_act for bio_act in bd.Database('biosphere3').search('Carbon dioxide, fossil', limit=1000) if (
        (bio_act['name'] == 'Carbon dioxide, fossil')
        & (bio_act['categories'] == ('air',))
    )][0]

    new_co2_flow = act.new_exchange(
        input=co2_flow,
        name=co2_flow['name'],
        categories=co2_flow['categories'],
        amount=amount,
        type='biosphere',
    )
    new_co2_flow['comment'] = "Added flow. " + new_co2_flow.get('comment', "")
    new_co2_flow.save()

    act.save()


def add_carbon_capture_to_plant(
        self,
        db_name: str,
        plant_type: str,
        activity_name: str = None,
        activity_code: str = None,
        capture_ratio: float = 0.95,
) -> None:
    """
    Add a carbon capture process to a technology, and modifies its direct carbon dioxide emissions

    :param db_name: name of the LCI database
    :param activity_name: name of the activity to be changed (to use only if the name of the activity is unique in the
        database)
    :param plant_type: type of the activity to be changed. Can be 'cement', 'hydrogen', 'municipal solid waste',
        'synthetic natural gas', 'wood', 'hard coal', 'lignite', or 'natural gas'.
    :param activity_code: code of the activity to be changed
    :param capture_ratio: carbon capture ratio, i.e., direct carbon dioxide emissions reduction rate
    :return: None (changes are saved in the database)
    """

    if activity_name is not None:
        act = [i for i in bd.Database(db_name).search(activity_name, limit=1000) if (
            (activity_name == i.as_dict()['name'])
        )][0]
    elif activity_code is not None:
        act = bd.Database(db_name).get(activity_code)
    else:
        raise ValueError("Either 'activity_name' or 'activity_code' should be provided")

    # Changing the activity name
    act['name'] += ' with CCS'

    # Add comment
    act['comment'] = ("Added CCS process and adjusted direct carbon dioxide emissions accordingly. "
                      + act.get('comment', ""))

    # Reducing the direct carbon dioxide emissions
    amount_co2_captured = 0  # total amount of CO2 captured
    for exc in act.biosphere():
        if exc['name'] in ['Carbon dioxide, fossil', 'Carbon dioxide, non-fossil']:
            amount_co2_captured += exc['amount'] * capture_ratio
            exc['amount'] *= (1 - capture_ratio)
            exc['comment'] = (f"Multiplied CO2 flows by factor: {round((1 - capture_ratio), 3)}. "
                              + exc.get('comment', ""))
            exc.save()

    # add processes required for CCS to technosphere flows for the different plant types
    if plant_type == 'cement':
        ccs_product_name = 'carbon dioxide, captured at cement plant'
        ccs_activity_name = 'carbon dioxide, captured at cement production plant, using monoethanolamine'
    elif plant_type == 'hydrogen':
        ccs_product_name = 'carbon dioxide, captured at hydrogen production plant, pre, pipeline 200km, storage 1000m'
        ccs_activity_name = 'carbon dioxide, captured at hydrogen production plant, pre, pipeline 200km, storage 1000m'
    elif plant_type == 'municipal solid waste':
        ccs_product_name = 'carbon dioxide, captured and reused'
        ccs_activity_name = 'carbon dioxide, captured at municipal solid waste incineration plant, for subsequent reuse'
    elif plant_type == 'synthetic natural gas':
        ccs_product_name = (
            'carbon dioxide, captured at synthetic natural gas plant, post, 200km pipeline, storage '
            '1000m')
        ccs_activity_name = (
            'carbon dioxide, captured at synthetic natural gas plant, post, 200km pipeline, storage '
            '1000m')
    elif plant_type == 'wood':
        ccs_product_name = (
            'carbon dioxide, captured at wood burning power plant 20 MW post, pipeline 200km, storage '
            '1000m')
        ccs_activity_name = (
            'carbon dioxide, captured at wood burning power plant 20 MW post, pipeline 200km, storage '
            '1000m')
    elif plant_type == 'hard coal':
        ccs_product_name = (
            'carbon dioxide, captured from hard coal-fired power plant, post, pipeline 200km, storage '
            '1000m')
        ccs_activity_name = (
            'carbon dioxide, captured from hard coal-fired power plant, post, pipeline 200km, storage '
            '1000m')
    elif plant_type == 'lignite':
        ccs_product_name = 'carbon dioxide, captured from lignite, post, pipeline 200km, storage 1000m'
        ccs_activity_name = 'carbon dioxide, captured from lignite, post, pipeline 200km, storage 1000m'
    elif plant_type == 'natural gas':
        ccs_product_name = 'carbon dioxide, captured from natural gas, post, 200km pipeline, storage 1000m'
        ccs_activity_name = 'carbon dioxide, captured from natural gas, post, 200km pipeline, storage 1000m'
    else:
        raise ValueError(f"Unexpected plant type: {plant_type}. Should be 'cement', 'hydrogen', 'municipal solid "
                         f"waste', 'synthetic natural gas', 'wood', 'hard coal', 'lignite', or 'natural gas'.")

    act_loc = act['location']
    ccs_act_loc = self.change_location_activity(
        activity=ccs_activity_name,
        product=ccs_product_name,
        location=act_loc,
        database=db_name,
        technosphere_or_biosphere_db=self.main_database,
    )

    ccs_act = [i for i in bd.Database(db_name).search(ccs_activity_name, limit=1000) if (
        (ccs_activity_name == i.as_dict()['name'])
        & (ccs_product_name == i.as_dict()['reference product'])
        & (ccs_act_loc == i.as_dict()['location'])
    )][0]

    # add a new non-fossil elementary flow to the activity
    new_ccs_exc = act.new_exchange(
        input=ccs_act,
        amount=amount_co2_captured,
        type='technosphere',
    )
    new_ccs_exc['comment'] = 'Added carbon capture flow'
    new_ccs_exc.save()

    act.save()
