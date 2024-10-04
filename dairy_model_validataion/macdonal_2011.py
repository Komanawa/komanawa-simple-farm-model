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

from komanawa.simple_farm_model import SimpleDairyModel
from komanawa.simple_farm_model.base_simple_farm_model import month_len
from komanawa.simple_farm_model.simple_dairy_model import mj_per_kg_dm, monly_ms_prod, dry_cow_feed, lactating_feed

# from table 1.
stocking_rates = np.array((2.2, 2.7, 3.1, 3.7, 4.3))  # cows/ha
pg_prod = np.array((18048, 18050, 19484, 18538, 20394))  # kg DM
pg_consumed = np.array((12098, 13785, 14322, 15609, 16597))  # kg DM
conserved_pg = np.array((1257, 1078, 806, 306, 65))  # kg DM
silage_consumed = np.array((354, 543, 562, 812, 876))  # kg DM
purchased_sup = np.array((0, 0, 0, 506, 812))  # kg DM
gross_rev = np.array((3860, 4206, 4572, 4805, 5392))  # $/ha
feed_cost = np.array((222, 297, 305, 359, 421))  # todo be careful with this as it may exclude the silage consumed...
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

milk_sales = np.array((4029, 4397, 4690, 4825, 4992))

scaler = np.array([monly_ms_prod[m] for m in all_months])
scaler = scaler / scaler.sum()
use_month_prod = scaler * (total_ms_yeild / stocking_rates)[-1]
milk_prod = {m: u / month_len[m] for m, u in zip(all_months, use_month_prod)}
lactating_feed = {k: v * 0.9 for k, v in lactating_feed.items()}
# abandoned... lactating_feed = {k: v * 123.19 for k, v in milk_prod.items()}

use_dry_cow_feed = deepcopy(dry_cow_feed)
use_dry_cow_feed[6] = 0
use_dry_cow_feed[7] = 0


class NoRepsDM(SimpleDairyModel):
    feed_per_cow_replacement = {m: 0 for m in all_months}  # remove feed from replacements
    feed_per_cow_lactating = lactating_feed
    # kgms_per_lac_cow = milk_prod
    feed_per_cow_dry = use_dry_cow_feed  # remove feed from cows in June/July
    # homegrown_efficiency = 0.9
    # homegrown_store_efficiency = 0.9
    # todo


def run_farm_model():
    outdata = pd.DataFrame(index=stocking_rates)

    for stock_rate in stocking_rates:
        idx = np.where(stocking_rates == stock_rate)[0][0]
        use_pg = pg_curve * pg_prod[idx]
        use_pg = np.array([u / month_len[m] for u, m in zip(use_pg, all_months)])
        dm = NoRepsDM(all_months, istate=[0], pg=use_pg,
                      ifeed=[silage_consumed[idx]],
                      imoney=[0], sup_feed_cost=feedprice, product_price=milksolid_price,
                      opt_mode='optimised', monthly_input=True,
                      peak_lact_cow_per_ha=stock_rate, ncore_opt=1, logging_level=logging.CRITICAL,
                      cull_dry_step=None, cull_levels=None, dryoff_levels=None)
        dm.run_model()
        tprod = np.nansum(dm.model_prod)
        last_lact = dm.out_lactating_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
        last_dry = dm.out_dry_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
        last_feed = dm.model_feed[-1][0] / mj_per_kg_dm
        feed_import = dm.model_feed_imported.sum() / mj_per_kg_dm
        total_feed_cost = dm.model_feed_cost.sum()
        outdata.loc[stock_rate, 'total_prod'] = tprod
        outdata.loc[stock_rate, 'prod_per_cow'] = tprod / stock_rate
        outdata.loc[stock_rate, 'last_lact'] = last_lact
        outdata.loc[stock_rate, 'last_dry'] = last_dry
        outdata.loc[stock_rate, 'last_feed'] = last_feed
        outdata.loc[stock_rate, 'feed_import'] = feed_import
        outdata.loc[stock_rate, 'total_feed_cost'] = total_feed_cost
        outdata.loc[stock_rate, 'feed_demand'] = np.nansum(dm.model_feed_demand) / mj_per_kg_dm
        outdata.loc[stock_rate, 'total_prod_money'] = np.nansum(dm.model_prod_money)

    base_idx = 3
    base_stock_rate = stocking_rates[base_idx]
    keys = 'total_prod', 'last_feed', 'feed_import', 'last_lact', 'feed_demand', 'prod_per_cow'
    nice_names = 'Total production', 'Feed Stored', 'Feed Imported', 'Fraction lactating Cows (EOY)', 'Feed Demand', 'Production per cow'
    units = 'kg MS', 'kg DM', 'kg DM', 'fraction', 'kg DM', 'kg MS'
    reported = total_ms_yeild, conserved_pg, purchased_sup, nanv, total_feed_demand, total_ms_yeild / stocking_rates
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 1, 1])
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis('off')
    axs = []
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r + 1, c])
            axs.append(ax)
    for key, rport, ax, nm, u in zip(keys, reported, axs, nice_names, units):
        got = outdata[key] / outdata.loc[base_stock_rate, key]
        expect = rport / rport[base_idx]
        if key != 'last_dry':
            ax.plot(stocking_rates, got, label=f'Modelled relative to {base_stock_rate} ' + '$cows~ha^{-1}$', c='r',
                    alpha=0.5)
        ax.plot(stocking_rates, expect, label=f'Macdonald (2011) relative to {base_stock_rate}' + ' $cows~ha^{-1}$',
                c='b',
                alpha=0.5)
        ax.set_ylabel(f'Relative to {base_stock_rate} cows/ha')

        got = outdata[key]
        expect = rport
        ax_raw = ax.twinx()
        ax_raw.set_ylabel(f'Absolute ({u})')
        ax_raw.plot(stocking_rates, got, label='Modelled', c='r', ls='--', alpha=0.5)
        ax_raw.plot(stocking_rates, expect, label='Macdonald (2011)', c='b', ls='--', alpha=0.5)
        ax.set_title(nm)

    hand, labs = ax.get_legend_handles_labels()
    hand_raw, labs_raw = ax_raw.get_legend_handles_labels()
    leg_ax.legend(hand + hand_raw, labs + labs_raw, loc='center', ncol=2)
    fig.supxlabel('Stocking rate (cows/ha)')

    fig.tight_layout()
    from pathlib import Path
    fig.savefig(Path.home().joinpath('Downloads', 'farm_model_validation_macdonald_2011.png'))
    plt.show()

    # todo plot stuff
    pass


if __name__ == '__main__':
    run_farm_model()
