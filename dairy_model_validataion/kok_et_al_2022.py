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
from komanawa.simple_farm_model.stock_rate_conversion import calc_full_farm_stock_rate
from op_expenses import get_operating_expenses
from supporting_script import _plot_outputs, _extract_data
from pathlib import Path

outdir = Path(__file__).parent.joinpath('Kok_2022_figs')
outdir.mkdir(exist_ok=True)

# validation against https://doi.org/10.33584/jnzg.2022.84.3567

year = [2018, 2018, 2018, 2019, 2019, 2019]
farm = ['LUDF', 'LSR', 'MSR', 'LUDF', 'LSR', 'MSR']
farm_uniq = ['LUDF-1', 'LSR-1', 'MSR-1', 'LUDF-2', 'LSR-2', 'MSR-2']
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

gross_rev = ms_prod * ms_price
opt_expese = np.array([5515, 5017, 7133, 6182, 5310, 7327])
net_income = gross_rev - opt_expese


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


class NoRepsDM(DairyModelWithSCScarcity):  # scarcity is necissary otherwise the model over imports...
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1


class NoRepsDMNoSupHandle(DairyModelWithSCScarcity):
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1
    sup_feedout_cost = 120 / 1000 / mj_per_kg_dm / 5
    homegrown_storage_cost = 175 / 1000 / mj_per_kg_dm / 5
    homegrown_store_efficiency = .85
    # sup_feedout_cost = 0
    # homegrown_storage_cost = 0


class NoRepsNoEff(DairyModelWithSCScarcity):
    supplemental_efficiency = 1
    feed_store_trigger = 0
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    ndays_feed_import = 1
    sup_feedout_cost = 0


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
                             cull_levels=100, dryoff_levels=100,
                             s=s, a=a, b=b, c=1, allow_mitigation_delay=True)
            dm.run_model()
            if explore_plot:
                fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                plt.show()
            _extract_data(dm, outdata, i, stock_rate,
                          add_sup_cost=base_feed_price * total_sup[i] * mj_per_kg_dm, )
    use_sup = total_sup
    xticks = [f'{f}-{y - 2000} {r}' + ' $c.ha^{-1}$' for y, f, r in
              zip(np.array(year)[xs], np.array(farm)[xs], stocking_rates[xs])]
    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates, use_sup, ms_prod, expect_feed_demand,
                                                  xs, gross_rev, net_income, opt_expese,
                                                  study_name='Kok et al. (2022)',
                                                  rel_lab='LUDF 2018,',
                                                  xticklabs=xticks,
                                                  base_x=0,
                                                  plot_rel=False, plot_money_rel=True)

    fig.suptitle('Feed passed as Supplement')
    fig_money.suptitle('Feed passed as Supplement')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('00_feed_passed_as_supplement.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('00_feed_passed_as_supplement_money.png'), dpi=300)


def make_data_silage_ifeed(explore_plot=False):
    base_feed_price = 400 / 1000 / mj_per_kg_dm
    outdata = pd.DataFrame(index=range(len((stocking_rates))))
    outdata['year'] = year
    outdata['farm'] = farm
    outdata['stocking_rate'] = stocking_rates
    a = 1
    b = 10
    s = 20

    for n in range(1):
        for i, stock_rate in enumerate(stocking_rates):
            year0 = year[i]
            farm0 = farm[i]
            use_pg = get_pgr(year0, farm0)
            if n == 0:
                ifeed_adder = 0
            else:
                ifeed_adder = outdata.loc[i, 'last_feed'] * mj_per_kg_dm

            ifeed = silage_supplement[i] * mj_per_kg_dm + ifeed_adder

            ifeed_based_on_dif = {'LUDF': 1000, 'LSR': 650,
                                  'MSR': 1000}
            cs = {'LUDF': 7, 'LSR': 5, 'MSR': 15}
            dm = NoRepsDMNoSupHandle(all_months, istate=[0], pg=use_pg,
                                     ifeed=[ifeed + ifeed_based_on_dif[farm0] * stock_rate * mj_per_kg_dm],
                                     # ifeed based on 4 rounds of 1500 kg DM/ha
                                     imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                                     monthly_input=True,
                                     peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                                     cull_dry_step=None, opt_mode='coarse',
                                     cull_levels=100, dryoff_levels=100,
                                     s=s, a=a, b=b, c=cs[farm0])
            dm.run_model()
            if explore_plot:
                fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                plt.show()
            _extract_data(dm, outdata, i, stock_rate, additional_stock_modifier=1)
    use_sup = barly_supp
    xticks = [f'{f}-{y - 2000} {r}' + ' $c.ha^{-1}$' for y, f, r in
              zip(np.array(year)[xs], np.array(farm)[xs], stocking_rates[xs])]

    export_data = pd.DataFrame(index=farm_uniq)
    export_data['prod_mod'] = outdata['total_prod'].values
    export_data['prod_mes'] = ms_prod
    export_data['feed_d_mod'] = outdata['total_feed_demand'].values
    export_data['feed_d_mes'] = expect_feed_demand
    export_data['feed_i_mod'] = outdata['feed_import'].values
    export_data['feed_i_mes'] = use_sup
    export_data['gross_mod'] = outdata['total_prod_money'].values
    export_data['gross_mes'] = gross_rev
    export_data['exp_mod'] = outdata['total_expenses'].values
    export_data['exp_mes'] = opt_expese
    export_data['net_mod'] = outdata['net_income'].values
    export_data['net_mes'] = net_income

    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates, use_sup, ms_prod, expect_feed_demand,
                                                  xs, gross_rev, net_income, opt_expese,
                                                  study_name='Kok et al. (2022)',
                                                  rel_lab='LUDF 2018,',
                                                  xticklabs=xticks,
                                                  base_x=0,
                                                  plot_rel=False, plot_money_rel=True)
    fig.suptitle('Pasture Based Dairy Platform')
    fig_money.suptitle('Pasture Based Dairy Platform')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('01_pasture_based_dairy_platform.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('01_pasture_based_dairy_platform_money.png'), dpi=300)

    return export_data


class ModFeedHandle(DairyModelWithSCScarcity):
    sup_feedout_cost = 120 / 1000 / mj_per_kg_dm / 4  # landed as 1/5 the cost as a start measure
    homegrown_storage_cost = 175 / 1000 / mj_per_kg_dm / 4  # landed as 1/5 the cost as a start measure
    homegrown_store_efficiency = .85  # increased the efficiency of the storage


def unmodified_comparison(explore_plot=False):
    base_feed_price = 400 / 1000 / mj_per_kg_dm
    outdata = pd.DataFrame(index=range(len((stocking_rates))))
    outdata_raw = pd.DataFrame(index=range(len((stocking_rates))))
    outdata['year'] = year
    outdata_raw['year'] = year
    outdata['farm'] = farm
    outdata_raw['farm'] = farm
    outdata['stocking_rate'] = stocking_rates
    outdata_raw['stocking_rate'] = stocking_rates
    a = 1
    b = 10
    s = 20

    for n in range(1):
        for i, stock_rate in enumerate(stocking_rates):
            year0 = year[i]
            farm0 = farm[i]
            use_pg = get_pgr(year0, farm0)
            if n == 0:
                ifeed_adder = 0
            else:
                ifeed_adder = outdata.loc[i, 'last_feed'] * mj_per_kg_dm

            ifeed = silage_supplement[i] * mj_per_kg_dm + ifeed_adder

            ifeed_based_on_dif = {'LUDF': 1000, 'LSR': 650, 'MSR': 1000}
            if year0 == 2018:
                cs = {'LUDF': 4, 'LSR': 1, 'MSR': 19}
            else:
                cs = {'LUDF': 8, 'LSR': 3, 'MSR': 19}

            # modify stock rate
            use_stock_rate, land_modifyer, cow_modifyer = calc_full_farm_stock_rate(stock_rate)
            ifeed_in = [ifeed + ifeed_based_on_dif[farm0] * use_stock_rate * mj_per_kg_dm]
            # ifeed_in = [0]

            dm = ModFeedHandle(all_months, istate=[0], pg=use_pg,
                               ifeed=ifeed_in,
                               # ifeed based on 4 rounds of 1500 kg DM/ha
                               imoney=[0], sup_feed_cost=base_feed_price, product_price=ms_price,
                               monthly_input=True,
                               peak_lact_cow_per_ha=use_stock_rate, ncore_opt=1,
                               logging_level=logging.CRITICAL,
                               cull_dry_step=None, opt_mode='coarse',
                               cull_levels=100, dryoff_levels=100,
                               s=s, a=a, b=b, c=cs[farm0])
            dm.run_model()
            if explore_plot:
                fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
                fig.suptitle(f'{year0} {farm0} {stock_rate} cows/ha')
                plt.show()
            _extract_data(dm, outdata, i, stock_rate, cow_modifer=cow_modifyer,
                          land_modifer=land_modifyer,
                          additional_stock_modifier=1)
            _extract_data(dm, outdata_raw, i, stock_rate,
                          additional_stock_modifier=1)
        print(outdata['last_feed'])

    use_sup = barly_supp

    export_data = pd.DataFrame(index=farm_uniq)
    export_data['prod_mod'] = outdata['total_prod'].values
    export_data['prod_mes'] = ms_prod
    export_data['feed_d_mod'] = outdata['total_feed_demand'].values
    export_data['feed_d_mes'] = expect_feed_demand
    export_data['feed_i_mod'] = outdata['feed_import'].values
    export_data['feed_i_mes'] = use_sup
    export_data['gross_mod'] = outdata['total_prod_money'].values
    export_data['gross_mes'] = gross_rev
    export_data['exp_mod'] = outdata['total_expenses'].values
    export_data['exp_mes'] = opt_expese
    export_data['net_mod'] = outdata['net_income'].values
    export_data['net_mes'] = net_income

    xticks = [f'{f}-{y - 2000} {r}' + ' $c.ha^{-1}$' for y, f, r in
              zip(np.array(year)[xs], np.array(farm)[xs], stocking_rates[xs])]
    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates, use_sup, ms_prod, expect_feed_demand,
                                                  xs, gross_rev, net_income, opt_expese,
                                                  study_name='Kok et al. (2022)',
                                                  rel_lab='LUDF 2018,',
                                                  xticklabs=xticks,
                                                  base_x=0,
                                                  plot_rel=False, plot_money_rel=True)
    fig.suptitle('Full System\nAdjusted for dairy platform size')
    fig_money.suptitle('Full System\nAdjusted for dairy platform size')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('02_adj_full_system.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('02_adj_full_system_money.png'), dpi=300)

    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata_raw,
                                                  stocking_rates, use_sup, ms_prod, expect_feed_demand,
                                                  xs, gross_rev, net_income, opt_expese,
                                                  study_name='Kok et al. (2022)',
                                                  rel_lab='LUDF 2018,',
                                                  xticklabs=xticks,
                                                  base_x=0,
                                                  plot_rel=False, plot_money_rel=True)
    fig.suptitle('Full System\nRaw values')
    fig_money.suptitle('Full System\nRaw values')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('02_raw_full_system.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('02_raw_full_system_money.png'), dpi=300)

    return export_data


def plot_pgr():
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'MSR': 'r', 'LSR': 'b', 'LUDF': 'orange'}
    lss = {2018: '-', 2019: ':'}
    for farm0, year0 in zip(farm, year):
        ax.plot(range(len(all_months)), get_pgr(year0, farm0), label=f'{farm0}-{year0}', color=colors[farm0],
                ls=lss[year0], alpha=0.75)
    ax.legend()
    ax.set_ylabel('Pasture Growth Rate ($kgDM~ha^{-1}day^{-1}$)')
    ax.set_xlabel('Month')
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('03_pgr.png'), dpi=300)


if __name__ == '__main__':
    plot_pgr()
    unmodified_comparison()
    make_data_no_pg_system()
    make_data_silage_ifeed()
