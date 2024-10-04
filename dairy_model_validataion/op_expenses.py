"""
created matt_dumont 
on: 10/5/24
"""
import numpy as np
def get_operating_expenses(peak_cow, distribution=False):
    """
    get the operating expenses for a farm
    :param distribution: bool if True return a dictionary of normal distributions for each expense, if False return the mean
    :return: Dictionary of operating expenses keys=['labour', 'stock', 'other', 'overheads']
    """
    # andrew provided the mean of north canterbury expenses 2019-2023,
    # email Fri, 16 Aug, 13:03, farm model optimisation (Zeb ignore)
    # https://mail.google.com/mail/u/0/#search/andrew%40primaryinsight.co.nz/KtbxLxgVTgHDmzlmvwwRvNXXmbrPfgQZnV
    # the base prices are calculated for the dairy platform only.
    # keynote scaled these according to Effect of stocking rate on the economics of pasture-based dairy farms
    # keynote scaled to peak cow
    full_farm_mult = peak_cow / 3.48  # 3.5 stocking density on platform and 2.82 on the support block
    fraction_ha = {  # fraction of the expense that is static per ha (vs variable per stocking density)
        'depreciation': 0.5,
        'rates': 1,
        'acc_levey': 1,
        'insurance': 1,
        'administration': .8,
        'freight': 0.05,
        'repairs_maintenance': 0.5,
        'vehicle': 0.35,
        'weed_pest': 0.9,
        'pasture_maintenance': 0.85,
        'irr_fert': 1,
        'electricity': 0.85,
        'animal_health': 0.9,
        'breeding_herd': 0.95,
        'dairy_shed': 0.25,
        'labour': 0.15,

    }

    means_v2 = {
        'depreciation': 718.3,
        'rates': 94.2,
        'acc_levey': 35.6,
        'insurance': 112.0,
        'administration': 197,
        'freight': 91.5,
        'repairs_maintenance': 240.2 + 378.8,
        'vehicle': 227.694020734657,
        'weed_pest': 43.8,
        'pasture_maintenance': 119.2,
        'irr_fert': 1343.8,
        'electricity': 146.4,
        'animal_health': 400,
        'breeding_herd': 253,
        'dairy_shed': 97,
        'labour': 1506.029,

    }

    means = {
        'labour': 1506.02947188769,  # is this somewhat included of other included expenses (like cost to feed out?)
        'stock': 897.178289424127,  # vet, etc.
        # 'feed':2852.46021932642, in the model, weird mix...
        'other': 2444.95668629038,  # N, irrigation, repairs maintenance...
        'overheads': 1157.08195177088,  # insurance, admin, rates

    }
    stdev_v2 = {

        'depreciation': 41.8,
        'rates': 2.5,
        'acc_levey': 2.3,
        'insurance': 13,
        'administration': 7.2,
        'freight': 3.2,
        'repairs_maintenance': 71,
        'vehicle': 36.6,
        'weed_pest': 13.5,
        'pasture_maintenance': 15,
        'irr_fert': 52.39,
        'electricity': 14.4,
        'animal_health': 7.4,
        'breeding_herd': 8.5,
        'dairy_shed': 3.9,
        'labour': 88.50
    }

    assert np.isclose(means['labour'], means_v2['labour'], 1)
    expect = means['stock']
    got = sum([means_v2[k] for k in ['animal_health',
                                     'breeding_herd',
                                     'dairy_shed']])
    assert np.isclose(expect, got, 1), f'dif: {expect - got=}'

    expect = means['other']
    got = sum([means_v2[k] for k in ['repairs_maintenance',
                                        'vehicle',
                                        'weed_pest',
                                        'pasture_maintenance',
                                        'irr_fert', ]])
    assert np.isclose(expect, got, 1), f'dif: {expect - got=}'

    expect = means['overheads']
    got = sum([means_v2[k] for k in ['depreciation',
                                     'rates',
                                     'acc_levey',
                                     'insurance',
                                     'administration']])
    assert np.isclose(expect, got, 1), f'dif: {expect - got=}'

    stds = {
        'labour': 88.5023741338805,
        'stock': 12.2681589905799,
        # 'feed':285.45276389778, in the model
        'other': 166.147557366323,
        'overheads': 43.3763029519386,

    }
    means_out =  {}
    for k, v in means_v2.items():
        means_out[k] = v * fraction_ha[k] + v * (1-fraction_ha[k]) * full_farm_mult

    if distribution:
        raise NotImplementedError
    else:
        return means_out

