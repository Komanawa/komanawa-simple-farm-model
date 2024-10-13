"""
created matt_dumont 
on: 10/12/24
"""
from datetime import datetime

import pandas as pd
from pathlib import Path
import numpy as np

import logging
from copy import deepcopy
import matplotlib.pyplot as plt
from macdonal_2011 import all_months
from komanawa.simple_farm_model import SimpleDairyModel, DairyModelWithSCScarcity
from komanawa.simple_farm_model.base_simple_farm_model import month_len
from komanawa.simple_farm_model.simple_dairy_model import mj_per_kg_dm, monly_ms_prod, dry_cow_feed, lactating_feed
from komanawa.simple_farm_model.stock_rate_conversion import calc_full_farm_stock_rate
from op_expenses import get_operating_expenses
from supporting_script import _plot_outputs, _extract_data

use_dry_cow_feed = deepcopy(dry_cow_feed)
use_dry_cow_feed[6] = 0
use_dry_cow_feed[7] = 0
mp_stocking_rate = 3.5
ms_price = 6.75
base_feed_price = 400 / 1000 / mj_per_kg_dm


def read_data():  # todo
    datadir = Path(__file__).parent
    paths = {
        2021: 'LUDF-2022-Data-Sheet-30-05-2022.csv',
        2022: 'LUDF-Weekly-Monitoring-Sheet-2022-2023.csv',
        2023: 'LUDF-Weekly-Monitoring-Sheet-2023-2024.csv'
    }
    outdata = {}
    outpg = []
    outall_months = []
    for y, p in paths.items():
        data = pd.read_csv(datadir.joinpath(p), skiprows=1, index_col=1)
        dates = [f'{d}-{y}' for d in data.columns[2:]]
        dates = pd.Series(pd.to_datetime(dates))
        dates[dates.dt.month <= 6] = dates[dates.dt.month <= 6] + pd.DateOffset(years=1)
        data.columns = ['dumm0', 'dumm1'] + list(dates)
        data = data[dates]

        temp_outdata = pd.DataFrame(index=dates)
        temp_outdata['area'] = area = pd.to_numeric(
            data.loc['Grazing ha available to milkers']) + pd.to_numeric(data.loc['Re-grassing'])
        t = (pd.to_numeric(data.loc['Total Cows Milk '])
             + pd.to_numeric(data.loc['Once-a-day milking cows'])
             + pd.to_numeric(data.loc['Lame cows current'])
             + pd.to_numeric(data.loc['Dry cows on farm'])
             )
        t = t.sum(axis=0)
        temp_outdata['cows'] = t / area
        milkers = pd.to_numeric(data.loc['Total Cows Milk ',])
        temp_outdata['pgr'] = pd.to_numeric(data.loc['Pasture Growth (pasture coach)'])
        temp_outdata['pasture_feed'] = pd.to_numeric(data.loc['Current Pasture Fed - Milkers']) * milkers / area
        temp_outdata['supplement_feed'] = pd.to_numeric(data.loc['Current Silage Fed - Milkers']) * milkers / area
        temp_outdata['feed_demand'] = temp_outdata['pasture_feed'] + temp_outdata['supplement_feed']
        temp_outdata['production'] = pd.to_numeric(data.loc['Per cow Production']) * milkers / area
        temp_outdata['sr'] = temp_outdata['cows'] / (temp_outdata['area'] / 4)
        temp_outdata['pgr_filled'] = temp_outdata['pgr'].interpolate()
        t = temp_outdata.groupby(temp_outdata.index.month)['pgr_filled'].mean().to_dict()
        t[6] = 20
        t[7] = 20
        t[8] = 20
        outpg.append([t[m] for m in all_months])
        outall_months.append(all_months)

        outdata[y] = temp_outdata
    outpg = np.concatenate(outpg)
    outall_months = np.concatenate(outall_months)
    return outdata, outpg, outall_months


class NoRepsDM(DairyModelWithSCScarcity):
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1


def dairy_platform_run(explore_plot):  # todo
    a = 1
    b = 5
    s = 10
    c = 20
    s, a, b, c = (18.445, 1.611, 0.726, 26.792)
    alldata, use_pg, use_all_months = read_data()
    ifeed = 600 * mj_per_kg_dm

    dm = NoRepsDM(use_all_months, istate=[0], pg=use_pg,
                  ifeed=[ifeed],
                  # ifeed based on 4 rounds of 1500 kg DM/ha
                  imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                  monthly_input=True,
                  peak_lact_cow_per_ha=mp_stocking_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                  cull_dry_step=None, opt_mode='coarse',
                  cull_levels=100, dryoff_levels=100,
                  s=s, a=a, b=b, c=20)
    dm.run_model()
    dm.long_name_dict['per_feed_import'] = 'Feed Import Percent'
    dm.obj_dict['per_feed_import'] = 'per_feed_import'
    dm.per_feed_import = dm.cum_feed_import / dm.get_annual_feed() * 100

    dm.long_name_dict['cum_feed_demand'] = 'Cumulative Feed Demand'
    dm.obj_dict['cum_feed_demand'] = 'cum_feed_demand'
    dm.cum_feed_demand = np.nancumsum(dm.model_feed_demand, axis=0)
    dm.long_name_dict['cum_prod'] = 'Cumulative Production'
    dm.obj_dict['cum_prod'] = 'cum_prod'
    dm.cum_prod = np.nancumsum(dm.model_prod, axis=0)

    if explore_plot:
        stitle = (f'Dairy Platform Only {mp_stocking_rate} '
                + '$cows~ha^{-1}$')
        fig, axs = dm.plot_results('feed', 'per_feed_import', 'cum_feed_import', c='r', start_year=2021)
        fig.suptitle(stitle)
        fig, axs = dm.plot_results('lactating_cow_fraction', 'feed_dry', 'feed_lactating', c='r', start_year=2021)
        fig.suptitle(stitle)

        fig, axs = dm.plot_results('cum_prod', 'cum_feed_import', 'cum_feed_demand', c='r', start_year=2021)
        fig.suptitle(stitle)
        data = pd.concat(list(alldata.values()))
        data = data.dropna()
        t = (data['production'] * 7).cumsum()
        use_x = data.index
        axs[0].plot(use_x, t, color='b')
        t2 = (data['feed_demand'] * 7 * mj_per_kg_dm).cumsum()
        axs[2].plot(use_x, t2, color='b')

        for y, data in alldata.items():
            axs[1].plot(data.index, (data['supplement_feed'] * 7 * mj_per_kg_dm).cumsum(), color='b')
        plt.show()



def raw_full_farm_run(explore_plot):  # todo
    s, a, b, c = (18.445, 1.611, 0.726, 26.792)
    alldata, use_pg, use_all_months = read_data()
    use_stock_rate, land_modifyer, cow_modifyer = calc_full_farm_stock_rate(mp_stocking_rate)
    ifeed = 600 * mj_per_kg_dm * land_modifyer

    dm = DairyModelWithSCScarcity(use_all_months, istate=[0], pg=use_pg,
                                  ifeed=[ifeed],
                                  # ifeed based on 4 rounds of 1500 kg DM/ha
                                  imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                                  monthly_input=True,
                                  peak_lact_cow_per_ha=use_stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                                  cull_dry_step=None, opt_mode='coarse',
                                  cull_levels=100, dryoff_levels=100,
                                  s=s, a=a, b=b, c=20)
    dm.run_model()
    dm.long_name_dict['per_feed_import'] = 'Feed Import Percent'
    dm.obj_dict['per_feed_import'] = 'per_feed_import'
    dm.per_feed_import = dm.cum_feed_import / dm.get_annual_feed() * 100

    dm.long_name_dict['cum_feed_demand'] = 'Cumulative Feed Demand'
    dm.obj_dict['cum_feed_demand'] = 'cum_feed_demand'
    dm.cum_feed_demand = np.nancumsum(dm.model_feed_demand, axis=0)
    dm.long_name_dict['cum_prod'] = 'Cumulative Production'
    dm.obj_dict['cum_prod'] = 'cum_prod'
    dm.cum_prod = np.nancumsum(dm.model_prod, axis=0)

    if explore_plot:
        stitle = (
                f'Full Model {round(use_stock_rate, 1)} '
                + '$cows~ha^{-1}$'
                + f' proxy for milking platform {mp_stocking_rate} '
                + '$cows~ha^{-1}$')
        fig, axs = dm.plot_results('feed', 'per_feed_import', 'cum_feed_import', c='r', start_year=2021)
        fig.suptitle(stitle)
        fig, axs = dm.plot_results('lactating_cow_fraction', 'feed_dry', 'feed_lactating', c='r', start_year=2021)
        fig.suptitle(stitle)

        fig, axs = dm.plot_results('cum_prod', 'cum_feed_import', 'cum_feed_demand', c='r', start_year=2021)
        fig.suptitle(stitle)
        data = pd.concat(list(alldata.values()))
        data = data.dropna()
        t = (data['production'] * 7 * land_modifyer).cumsum()
        use_x = data.index
        axs[0].plot(use_x, t, color='b')
        t2 = (data['feed_demand'] * 7 * mj_per_kg_dm * land_modifyer).cumsum()
        axs[2].plot(use_x, t2, color='b')

        for y, data in alldata.items():
            axs[1].plot(data.index, (data['supplement_feed'] * 7 * mj_per_kg_dm * land_modifyer).cumsum(), color='b')
        plt.show()



if __name__ == '__main__':
    dairy_platform_run(True)  # todo looking good I just need to clean up the plot.
    raw_full_farm_run(True)  # todo looking good I just need to clean up the plot and save...
