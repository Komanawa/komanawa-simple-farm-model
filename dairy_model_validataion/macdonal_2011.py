"""

Farm model validataion against macdonald 2011
Note this dairy system does not support replacments in the model
DOI: 10.3168/jds.2010-3688
URL: https://www.sciencedirect.com/science/article/pii/S0022030211002359
created matt_dumont 
on: 10/3/24
"""
import logging
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from komanawa.simple_farm_model import SimpleDairyModel, DairyModelWithSCScarcity
from komanawa.simple_farm_model.base_simple_farm_model import month_len
from komanawa.simple_farm_model.simple_dairy_model import mj_per_kg_dm, monly_ms_prod, dry_cow_feed, lactating_feed
from komanawa.simple_farm_model.stock_rate_conversion import calc_full_farm_stock_rate
from supporting_script import _extract_data, _plot_outputs
from pathlib import Path

outdir = Path(__file__).parent.joinpath('Macdonald_2011_figs')
outdir.mkdir(exist_ok=True)

# from table 1.
stocking_rates = np.array((2.2, 2.7, 3.1, 3.7, 4.3))  # cows/ha
pg_prod = np.array((18048, 18050, 19484, 18538, 20394))  # kg DM
pg_consumed = np.array((12098, 13785, 14322, 15609, 16597))  # kg DM
conserved_pg = np.array((1257, 1078, 806, 306, 65))  # kg DM
silage_consumed = np.array((354, 543, 562, 812, 876))  # kg DM
purchased_sup = np.array((0, 0, 0, 506, 812))  # kg DM
gross_rev = np.array((3860, 4206, 4572, 4805, 5392))  # $/ha
feed_cost = np.array((222, 297, 305, 359, 421))  # be careful with this as it may exclude the silage consumed...
feedprice = 0.3 / mj_per_kg_dm  # $/kg MJ
total_feed_demand = pg_consumed + silage_consumed + purchased_sup
total_sup = silage_consumed + purchased_sup
feed_cost_use = purchased_sup * mj_per_kg_dm * feedprice

milkfat_yild = np.array((507, 557, 595, 625, 647))
milkprotien_yild = np.array((388, 415, 452, 467, 494))
lactose_yeild = np.array((537, 570, 626, 653, 723))
total_ms_yeild = milkfat_yild + milkprotien_yild + lactose_yeild
nanv = total_ms_yeild * np.nan
# values from https://www.dairynz.co.nz/media/m4knoxds/facts_and_figures_chapter_4_updated_december_2021.pdf

annual_ms_prod = np.array([250, 300, 350, 400, 450, 500, 550, 600])
annual_me_req = np.array([4.1, 4.5, 4.8, 5.2, 5.5, 5.9, 6.2, 6.7]) * 1000 * mj_per_kg_dm

expected_me = np.interp(total_ms_yeild, annual_ms_prod, annual_me_req)
me_perms = annual_me_req / annual_ms_prod
milksolid_price = np.mean(gross_rev / total_ms_yeild)  # 4.30  # paper reports 4.30, but this excludes lactose # $/kg
milksolid_price = 4.3
all_months = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
pg_curve = np.array([26, 35, 64, 83, 87, 65, 67, 42, 34, 25, 29, 22, ])
pg_curve = pg_curve * np.array([month_len[m] for m in all_months])
pg_curve = pg_curve / pg_curve.sum()

opt_expenses = np.array([2732, 3090, 3311, 3633, 3993]) - feed_cost  # $/ha
expect_milk_prod_price = total_ms_yeild * milksolid_price
expect_net_income = expect_milk_prod_price - feed_cost_use - opt_expenses

use_dry_cow_feed = deepcopy(dry_cow_feed)
use_dry_cow_feed[6] = 0
use_dry_cow_feed[7] = 0

use_lact_feed = {k: v * 0.9 for k, v in lactating_feed.items()}


class NoRepsDM(DairyModelWithSCScarcity):
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_lactating = use_lact_feed
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July


class LowerFeedDMWSCS(DairyModelWithSCScarcity):
    feed_per_cow_lactating = use_lact_feed


def run_farm_model(explore_plot=False):
    outdata = pd.DataFrame(index=range(len(stocking_rates)))
    a = 1
    b = 10
    s = 20

    cs = {2.2: 5, 2.7: 5, 3.1: 5, 3.7: 8, 4.3: 9}
    for i, stock_rate in enumerate(stocking_rates):
        idx = np.where(stocking_rates == stock_rate)[0][0]
        use_pg = pg_curve * pg_prod[idx]
        use_pg = np.array([u / month_len[m] for u, m in zip(use_pg, all_months)])
        ifeed_in = [silage_consumed[idx] * mj_per_kg_dm + 500 * stock_rate * mj_per_kg_dm]
        dm = NoRepsDM(all_months, istate=[0], pg=use_pg,
                      ifeed=ifeed_in,
                      imoney=[0], sup_feed_cost=feedprice, product_price=milksolid_price,
                      monthly_input=True,
                      peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                      opt_mode='coarse',
                      cull_levels=100, dryoff_levels=100,
                      s=s, a=a, b=b, c=cs[stock_rate])
        dm.run_model()
        if explore_plot:
            fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
            fig.suptitle(f'{stock_rate} cows/ha')
            fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
            fig.suptitle(f'{stock_rate} cows/ha')
            plt.show()

        _extract_data(dm, outdata, i, stock_rate, land_modifer=1, cow_modifer=1, add_sup_cost=0,
                      additional_stock_modifier=1, expense_modifier=0.73)
    base_idx = 3
    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates,
                                                  use_sup=purchased_sup,
                                                  ms_prod=total_ms_yeild, expect_feed_demand=total_feed_demand,
                                                  xs=range(len(stocking_rates)),
                                                  gross_rev=expect_milk_prod_price,
                                                  net_income=expect_net_income, opt_expese=opt_expenses,
                                                  study_name='Macdonald (2011)',
                                                  rel_lab='',
                                                  base_x=base_idx,
                                                  xticklabs=[f'{sr}' + ' $cows~ha^{-1}$' for sr in stocking_rates],

                                                  plot_rel=False, plot_money_rel=False,
                                                  )
    fig.suptitle('Dairy Platform Only')
    fig_money.suptitle('Dairy Platform Only')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('00_dairy_platform_only.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('00_dairy_platform_only_money.png'), dpi=300)


def run_farm_model_no_mod(explore_plot=False):
    outdata = pd.DataFrame(index=range(len(stocking_rates)))
    outdata_raw = pd.DataFrame(index=range(len(stocking_rates)))
    a = 1
    b = 10
    s = 20

    cs = {2.2: 5, 2.7: 5, 3.1: 7, 3.7: 15, 4.3: 15}
    for i, stock_rate in enumerate(stocking_rates):
        idx = np.where(stocking_rates == stock_rate)[0][0]
        use_pg = pg_curve * pg_prod[idx]
        use_pg = np.array([u / month_len[m] for u, m in zip(use_pg, all_months)])
        use_stock_rate, land_modifyer, cow_modifyer = calc_full_farm_stock_rate(stock_rate)

        ifeed_in = [silage_consumed[idx] * mj_per_kg_dm + 500 * use_stock_rate * mj_per_kg_dm]

        dm = LowerFeedDMWSCS(all_months, istate=[0], pg=use_pg,
                             ifeed=ifeed_in,
                             imoney=[0], sup_feed_cost=feedprice, product_price=milksolid_price,
                             monthly_input=True,
                             peak_lact_cow_per_ha=use_stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                             opt_mode='coarse',
                             cull_levels=100, dryoff_levels=100,
                             s=s, a=a, b=b, c=cs[stock_rate])
        dm.run_model()
        if explore_plot:
            fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
            fig.suptitle(f'{stock_rate} cows/ha')
            fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
            fig.suptitle(f'{stock_rate} cows/ha')
            plt.show()

        _extract_data(dm, outdata, i, stock_rate,

                      cow_modifer=cow_modifyer, land_modifer=land_modifyer,
                      add_sup_cost=0,
                      additional_stock_modifier=1, expense_modifier=0.73)

        _extract_data(dm, outdata_raw, i, stock_rate,

                      # cow_modifer=cow_modifyer, land_modifer=land_modifyer,
                      add_sup_cost=0,
                      additional_stock_modifier=1, expense_modifier=0.73)
    base_idx = 3
    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates,
                                                  use_sup=purchased_sup,
                                                  ms_prod=total_ms_yeild, expect_feed_demand=total_feed_demand,
                                                  xs=range(len(stocking_rates)),
                                                  gross_rev=expect_milk_prod_price,
                                                  net_income=expect_net_income, opt_expese=opt_expenses,
                                                  study_name='Macdonald (2011)',
                                                  rel_lab='',
                                                  base_x=base_idx,
                                                  xticklabs=[f'{sr}' + ' $cows~ha^{-1}$' for sr in stocking_rates],

                                                  plot_rel=False, plot_money_rel=False
                                                  )
    fig.suptitle('Full System - Similar Imports\nAdjusted for dairy platform size')
    fig_money.suptitle('Full System - Similar Imports\nAdjusted for dairy platform size')
    fig.tight_layout()
    fig_money.tight_layout()

    fig.savefig(outdir.joinpath('01_adj_full_system_similar_imports.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('01_adj_full_system_similar_imports_money.png'), dpi=300)

    fig_raw, axs_raw, fig_money_raw, moneyaxs_raw = _plot_outputs(outdata_raw,
                                                                  stocking_rates,
                                                                  use_sup=purchased_sup,
                                                                  ms_prod=total_ms_yeild,
                                                                  expect_feed_demand=total_feed_demand,
                                                                  xs=range(len(stocking_rates)),
                                                                  gross_rev=expect_milk_prod_price,
                                                                  net_income=expect_net_income, opt_expese=opt_expenses,
                                                                  study_name='Macdonald (2011)',
                                                                  rel_lab='',
                                                                  base_x=base_idx,
                                                                  xticklabs=[f'{sr}' + ' $cows~ha^{-1}$' for sr in
                                                                             stocking_rates],

                                                                  plot_rel=True, plot_money_rel=True,

                                                                  )
    fig_raw.suptitle('Full System - Similar Imports\nRaw values')
    fig_money_raw.suptitle('Full System - Similar Imports\nRaw values')
    fig_raw.tight_layout()
    fig_money_raw.tight_layout()
    fig_raw.savefig(outdir.joinpath('01_raw_full_system_similar_imports.png'), dpi=300)
    fig_money_raw.savefig(outdir.joinpath('01_raw_full_system_similar_imports_money.png'), dpi=300)


def run_farm_model_no_mod_no_feed_limit(explore_plot=False):
    outdata = pd.DataFrame(index=range(len(stocking_rates)))
    outdata_raw = pd.DataFrame(index=range(len(stocking_rates)))
    a = 1
    b = 10
    s = 20

    cs = {2.2: 10, 2.7: 10, 3.1: 15, 3.7: 25, 4.3: 25}
    for i, stock_rate in enumerate(stocking_rates):
        idx = np.where(stocking_rates == stock_rate)[0][0]
        use_pg = pg_curve * pg_prod[idx]
        use_pg = np.array([u / month_len[m] for u, m in zip(use_pg, all_months)])
        use_stock_rate, land_modifyer, cow_modifyer = calc_full_farm_stock_rate(stock_rate)
        ifeed_in = [silage_consumed[idx] * mj_per_kg_dm + 500 * use_stock_rate * mj_per_kg_dm]

        dm = LowerFeedDMWSCS(all_months, istate=[0], pg=use_pg,
                             ifeed=ifeed_in,
                             imoney=[0], sup_feed_cost=feedprice, product_price=milksolid_price,
                             monthly_input=True,
                             peak_lact_cow_per_ha=use_stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                             opt_mode='coarse',
                             cull_levels=100, dryoff_levels=100,
                             s=s, a=a, b=b, c=cs[stock_rate])
        dm.run_model()
        if explore_plot:
            fig, axs = dm.plot_results('feed', 'lactating_cow_fraction', 'cum_feed_import')
            fig.suptitle(f'{stock_rate} cows/ha')
            fig, axs = dm.plot_results('feed_replacement', 'feed_dry', 'feed_lactating')
            fig.suptitle(f'{stock_rate} cows/ha')
            plt.show()

        _extract_data(dm, outdata, i, stock_rate,

                      cow_modifer=cow_modifyer, land_modifer=land_modifyer,
                      add_sup_cost=0,
                      additional_stock_modifier=1, expense_modifier=0.73)

        _extract_data(dm, outdata_raw, i, stock_rate,

                      # cow_modifer=cow_modifyer, land_modifer=land_modifyer,
                      add_sup_cost=0,
                      additional_stock_modifier=1, expense_modifier=0.73)
    base_idx = 3
    fig, axs, fig_money, moneyaxs = _plot_outputs(outdata,
                                                  stocking_rates,
                                                  use_sup=purchased_sup,
                                                  ms_prod=total_ms_yeild, expect_feed_demand=total_feed_demand,
                                                  xs=range(len(stocking_rates)),
                                                  gross_rev=expect_milk_prod_price,
                                                  net_income=expect_net_income, opt_expese=opt_expenses,
                                                  study_name='Macdonald (2011)',
                                                  rel_lab='',
                                                  base_x=base_idx,
                                                  xticklabs=[f'{sr}' + ' $cows~ha^{-1}$' for sr in stocking_rates],

                                                  plot_rel=True, plot_money_rel=True,
                                                  )
    fig.suptitle('Full System - Less Feed Constrained\nAdjusted for dairy platform size')
    fig_money.suptitle('Full System - Less Feed Constrained\nAdjusted for dairy platform size')
    fig.tight_layout()
    fig_money.tight_layout()
    fig.savefig(outdir.joinpath('02_adj_full_system_less_feed_constrained.png'), dpi=300)
    fig_money.savefig(outdir.joinpath('02_adj_full_system_less_feed_constrained_money.png'), dpi=300)

    fig_raw, axs_raw, fig_money_raw, moneyaxs_raw = _plot_outputs(outdata_raw,
                                                                  stocking_rates,
                                                                  use_sup=purchased_sup,
                                                                  ms_prod=total_ms_yeild,
                                                                  expect_feed_demand=total_feed_demand,
                                                                  xs=range(len(stocking_rates)),
                                                                  gross_rev=expect_milk_prod_price,
                                                                  net_income=expect_net_income, opt_expese=opt_expenses,
                                                                  study_name='Macdonald (2011)',
                                                                  rel_lab='',
                                                                  base_x=base_idx,
                                                                  xticklabs=[f'{sr}' + ' $cows~ha^{-1}$' for sr in
                                                                             stocking_rates],

                                                                  plot_rel=True, plot_money_rel=True,
                                                                  )
    fig_raw.suptitle('Full System - Less Feed Constrained\nRaw values')
    fig_money_raw.suptitle('Full System - Less Feed Constrained\nRaw values')
    fig_raw.tight_layout()
    fig_money_raw.tight_layout()
    fig_raw.savefig(outdir.joinpath('02_raw_full_system_less_feed_constrained.png'), dpi=300)
    fig_money_raw.savefig(outdir.joinpath('02_raw_full_system_less_feed_constrained_money.png'), dpi=300)


if __name__ == '__main__':
    run_farm_model_no_mod_no_feed_limit()
    run_farm_model_no_mod()
    run_farm_model()
