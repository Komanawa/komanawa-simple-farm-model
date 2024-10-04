"""
created matt_dumont 
on: 10/4/24
"""

import numpy as np

import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from komanawa.simple_farm_model import SimpleDairyModel, DairyModelWithSCScarcity
from komanawa.simple_farm_model.base_simple_farm_model import month_len
from komanawa.simple_farm_model.simple_dairy_model import mj_per_kg_dm, monly_ms_prod, dry_cow_feed, lactating_feed

# todo validation against https://doi.org/10.33584/jnzg.2022.84.3567

year = [2018, 2018, 2018, 2019, 2019, 2019]
farm = ['LUDF', 'LSR', 'MSR', 'LUDF', 'LSR', 'MSR']
xs = [0, 3, 1, 4, 2, 5]
stocking_rates = np.array([3.4, 2.9, 3.9, 3.4, 2.9, 3.9])
pasture_prod = np.array([19.7, 15.8, 16.5, 19.4, 16.6, 16.1]) * 1000  # kg DM/ha
ms_prod = np.array([1721, 1308, 1870, 1757, 1185, 1574])
silage_supplement = np.array([0.23, 0.37, 0.15, 0.49, 0.01, 0.02]) * 1000 * stocking_rates  # kg DM/ha
barly_supp = np.array(
    [0, 0, 0.54, 0, 0, 0.55]) * 1000 * 13.0 / 11 * stocking_rates  # convert from ton barley to kg DM equivalent
pasture_off = np.array([5.2, 4.2, 4.1, 5.1, 4.5, 3.8]) * 1000 * stocking_rates
total_sup = silage_supplement + barly_supp
all_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
expect_feed_demand = total_sup + pasture_off

pg_curve = np.array([20, 20, 34, 62.5, 82, 80, 70.5, 72, 59, 48, 37, 20])

pg_total = np.array([p * month_len[m] for m, p in zip(all_months, pg_curve)])
pg_curve = pg_total / pg_total.sum()

got_mj_per_ms = ((pasture_prod + silage_supplement + barly_supp) * mj_per_kg_dm) / ms_prod

use_dry_cow_feed = deepcopy(dry_cow_feed)
use_dry_cow_feed[6] = 0
use_dry_cow_feed[7] = 0
ms_price = 6.75


def get_pgr(year, site):
    assert site in farm
    assert year == 2018 or year == 2019, 'year must be 2018 or 2019'
    if year == 2018:
        base_path = Path(__file__).parent.joinpath('kok_et_al_2018-2019_pg.csv')
    else:
        base_path = Path(__file__).parent.joinpath('kok_et_al_2019-2020_pg.csv')

    data = pd.read_csv(base_path, header=[0, 1])
    index = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    data.index = index
    outdata = data[site]['Y']
    return outdata


class NoRepsDM(DairyModelWithSCScarcity):  # todo scaricty is necissary otherwise the model over imports...
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1
    # todo something feels weird... homegrown_store_efficiency = 0.9
    # todo something feels weird... homegrown_efficiency = .9
    # todo something feels weird... supplemental_efficiency = .95


class NoRepsNoEff(DairyModelWithSCScarcity):
    supplemental_efficiency = 1
    feed_store_trigger = 0
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1


def make_data_no_pg_system(explore_plot=False):
    a = 1
    b = 5
    s = 10

    base_feed_price = 400 / 1000 / mj_per_kg_dm
    outdata = pd.DataFrame(index=range(len((stocking_rates))))
    outdata['year'] = year
    outdata['farm'] = farm
    outdata['stocking_rate'] = stocking_rates

    for n in range(1):
        for i, stock_rate in enumerate(stocking_rates):
            year0 = year[i]
            farm0 = farm[i]
            if n == 0:
                ifeed_adder = 0
            else:
                ifeed_adder = outdata.loc[i, 'last_feed'] * mj_per_kg_dm
            use_pg = get_pgr(year0, farm0) * 0
            dm = NoRepsNoEff(all_months, istate=[0], pg=use_pg,
                             ifeed=[expect_feed_demand[i] * mj_per_kg_dm + ifeed_adder],
                             imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                             monthly_input=True,
                             peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                             cull_dry_step=None, opt_mode='coarse',
                             cull_levels=1000, dryoff_levels=50,
                             s=s, a=a, b=b, c=1, allow_mitigation_delay=True)
            dm.run_model()
            if explore_plot:
                fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                plt.show()
            tprod = np.nansum(dm.model_prod)
            last_lact = dm.out_lactating_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_dry = dm.out_dry_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_feed = dm.model_feed[-1][0] / mj_per_kg_dm
            feed_import = dm.model_feed_imported.sum() / mj_per_kg_dm
            total_feed_cost = dm.model_feed_cost.sum()
            outdata.loc[i, 'total_prod'] = tprod
            outdata.loc[i, 'prod_per_cow'] = tprod / stock_rate
            outdata.loc[i, 'last_lact'] = last_lact
            outdata.loc[i, 'last_dry'] = last_dry
            outdata.loc[i, 'last_feed'] = last_feed
            outdata.loc[i, 'feed_import'] = feed_import
            outdata.loc[i, 'total_feed_cost'] = total_feed_cost
            outdata.loc[i, 'feed_demand'] = np.nansum(dm.model_feed_demand) / mj_per_kg_dm
            outdata.loc[i, 'total_prod_money'] = np.nansum(dm.model_prod_money)
        print(outdata['last_feed'])

    fig, axs = _plot_outputs(outdata)
    plt.show()


def make_data_not_fixed_system(explore_plot=False):
    # todo move to s curve
    base_feed_price = 400 / 1000 / mj_per_kg_dm
    outdata = pd.DataFrame(index=range(len((stocking_rates))))
    outdata['year'] = year
    outdata['farm'] = farm
    outdata['stocking_rate'] = stocking_rates
    a = 1
    b = 1
    s = 10

    for n in range(1):
        for i, stock_rate in enumerate(stocking_rates):
            year0 = year[i]
            farm0 = farm[i]
            use_pg = get_pgr(year0, farm0)
            if n == 0:
                ifeed_adder = 0
            else:
                ifeed_adder = outdata.loc[i, 'last_feed'] * mj_per_kg_dm
            dm = NoRepsDM(all_months, istate=[0], pg=use_pg,
                          ifeed=[0 + ifeed_adder],
                          imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                          opt_mode='optimised', monthly_input=True,
                          peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                          cull_dry_step=None, cull_levels=None, dryoff_levels=None,
                          s=s, a=a, b=b, c=20)  # todo play with C...
            dm.run_model()
            if explore_plot:
                fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                plt.show()
            tprod = np.nansum(dm.model_prod)
            last_lact = dm.out_lactating_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_dry = dm.out_dry_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_feed = dm.model_feed[-1][0] / mj_per_kg_dm
            feed_import = dm.model_feed_imported.sum() / mj_per_kg_dm
            total_feed_cost = dm.model_feed_cost.sum()
            outdata.loc[i, 'total_prod'] = tprod
            outdata.loc[i, 'prod_per_cow'] = tprod / stock_rate
            outdata.loc[i, 'last_lact'] = last_lact
            outdata.loc[i, 'last_dry'] = last_dry
            outdata.loc[i, 'last_feed'] = last_feed
            outdata.loc[i, 'feed_import'] = feed_import
            outdata.loc[i, 'total_feed_cost'] = total_feed_cost
            outdata.loc[i, 'feed_demand'] = np.nansum(dm.model_feed_demand) / mj_per_kg_dm
            outdata.loc[i, 'total_prod_money'] = np.nansum(dm.model_prod_money)
        print(outdata['last_feed'])
    fig, axs = _plot_outputs(outdata)
    plt.show()
    pass


def make_data_fixed_import():
    # todo move to s curve
    base_feed_price = 400000000000 / 1000 / mj_per_kg_dm  # todo this does not work...
    outdata = pd.DataFrame(index=range(len((stocking_rates))))
    outdata['year'] = year
    outdata['farm'] = farm
    outdata['stocking_rate'] = stocking_rates

    for n in range(5):
        for i, stock_rate in enumerate(stocking_rates):
            year0 = year[i]
            farm0 = farm[i]
            use_pg = get_pgr(year0, farm0)
            if n == 0:
                ifeed_adder = 0
            else:
                ifeed_adder = outdata.loc[i, 'last_feed'] * mj_per_kg_dm
            dm = NoRepsDM(all_months, istate=[0], pg=use_pg,
                          ifeed=[total_sup[i] * mj_per_kg_dm + ifeed_adder],
                          imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                          opt_mode='optimised', monthly_input=True,
                          peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                          cull_dry_step=None, cull_levels=None, dryoff_levels=None)
            dm.run_model()
            fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
            fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
            fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
            fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
            plt.show()
            tprod = np.nansum(dm.model_prod)
            last_lact = dm.out_lactating_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_dry = dm.out_dry_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
            last_feed = dm.model_feed[-1][0] / mj_per_kg_dm
            feed_import = dm.model_feed_imported.sum() / mj_per_kg_dm
            total_feed_cost = dm.model_feed_cost.sum()
            outdata.loc[i, 'total_prod'] = tprod
            outdata.loc[i, 'prod_per_cow'] = tprod / stock_rate
            outdata.loc[i, 'last_lact'] = last_lact
            outdata.loc[i, 'last_dry'] = last_dry
            outdata.loc[i, 'last_feed'] = last_feed
            outdata.loc[i, 'feed_import'] = feed_import
            outdata.loc[i, 'total_feed_cost'] = total_feed_cost
            outdata.loc[i, 'feed_demand'] = np.nansum(dm.model_feed_demand) / mj_per_kg_dm
            outdata.loc[i, 'total_prod_money'] = np.nansum(dm.model_prod_money)
        print(outdata['last_feed'])
    fig, axs = _plot_outputs(outdata)
    plt.show()
    pass


def _plot_outputs(outdata, plot_rel=False):
    nanv = np.full_like(stocking_rates, np.nan)
    keys = 'total_prod', 'last_feed', 'feed_import', 'last_lact', 'feed_demand', 'prod_per_cow'
    nice_names = 'Total production', 'Feed Stored', 'Feed Imported', 'Fraction lactating Cows (EOY)', 'Feed Demand', 'Production per cow'
    units = 'kg MS', 'kg DM', 'kg DM', 'fraction', 'kg DM', 'kg MS'
    reported = ms_prod, nanv, total_sup, nanv, expect_feed_demand, ms_prod / stocking_rates
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 1, 1])
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis('off')
    axs = []
    xplot = np.arange(len(stocking_rates))
    xticks = [f'{y}-{f}\n{r}' + ' $cow~ha^{-1}$' for y, f, r in zip(
        np.array(year)[xs], np.array(farm)[xs], stocking_rates[xs])]
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r + 1, c])
            axs.append(ax)
    for key, rport, ax, nm, u in zip(keys, reported, axs, nice_names, units):
        got = outdata.loc[xs, key] / outdata.loc[0, key]
        expect = rport[xs] / rport[0]

        if plot_rel:
            ax_raw = ax.twinx()
            if key != 'last_dry':
                ax_raw.plot(xplot, got, label=f'Modelled relative to LUDF 2018, 3.5 ' + '$cows~ha^{-1}$', c='r',
                            alpha=0.5, ls='--', )
            ax_raw.plot(xplot, expect, label=f'Kok et al. (2022) relative to LUDF 2018, 3.5' + ' $cows~ha^{-1}$',
                        c='b', ls='--',
                        alpha=0.5)
        ax.set_ylabel(f'Relative to LUDF 2018, 3.5' + ' $cows~ha^{-1}$')
        ax.set_xticks(xplot)
        ax.set_xticklabels(xticks, rotation=-30)

        got = outdata.loc[xs, key]
        expect = rport[xs]
        ax.set_ylabel(f'Absolute ({u})')
        ax.plot(xplot, got, label='Modelled', c='r', alpha=0.5, marker='o')
        ax.plot(xplot, expect, label='Kok et al. (2022)', c='b', alpha=0.5, marker='o')
        ax.set_title(nm)

    hand, labs = ax.get_legend_handles_labels()
    if plot_rel:
        hand_raw, labs_raw = ax_raw.get_legend_handles_labels()
    else:
        hand_raw, labs_raw = [], []
    leg_ax.legend(hand + hand_raw, labs + labs_raw, loc='center', ncol=2)
    fig.supxlabel('Stocking rate (cows/ha)')

    fig.tight_layout()
    return fig, axs


# todo a version that limits the pgr?

if __name__ == '__main__':
    t = get_pgr(2019, 'LUDF')
    t = sum([t*month_len[m] for t,m in zip(t,all_months)])
    print(t)
    make_data_not_fixed_system() # todo something weird is happening here not sure what...
    make_data_no_pg_system()  # todo looking pretty good
    # make_data_fixed_import()
