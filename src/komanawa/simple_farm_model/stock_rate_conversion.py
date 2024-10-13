"""
created matt_dumont 
on: 10/8/24
"""
import numpy as np


def calc_full_farm_stock_rate(milk_platform_sr):
    """
    calculate the peak cow (peak lactating cow /ha) for a full farm from the milk platform stocking rate.

    :param milk_platform_sr:
    :return:
    """
    cow_modifer = 1.44
    single_val = True
    if hasattr(milk_platform_sr, '__iter__'):
        single_val = False
    milk_platform_sr = np.atleast_1d(milk_platform_sr)
    overwinter_stock = 2 / 12 * milk_platform_sr
    additional_stock = ((cow_modifer - 1) * milk_platform_sr) + overwinter_stock
    additional_land = additional_stock / (8.5 / 5 * milk_platform_sr)
    total_stock = milk_platform_sr * cow_modifer
    use_stock_rate = milk_platform_sr / (1 + additional_land)
    use_stock_rate = use_stock_rate.round(3)
    land_modifier = 1 / (1 + additional_land)
    if single_val:
        return use_stock_rate[0], land_modifier[0], cow_modifer
    return use_stock_rate, land_modifier, np.full_like(milk_platform_sr, cow_modifer)
