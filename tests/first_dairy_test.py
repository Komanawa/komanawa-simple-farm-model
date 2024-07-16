"""
created matt_dumont 
on: 6/27/24
"""

# todo eventually add this to test_simple_dairy_model.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from komanawa.simple_farm_model.simple_dairy_model import SimpleDairyModel, month_len, DairyModelWithScarcity


# todo need to make feed cost a non-linear relationship with feed demand... how to justify this!?, does ryan have this???
# todo some sort of s-curve relationship with feed demand and feed cost... how to justify
#  The test model works really well with scarcity cost.  Just need to justify the s curve...

def test_average_year():
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = 8.09

    print(f'sup_cost (kgms): {sup_cost * 123.19}, product_price: {product_price}')

    data_path = Path(__file__).parent.joinpath('test_data', 'baseline-mod-median.csv')
    pg_data = pd.read_csv(data_path, index_col=0)['0']
    all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
    pg1 = [pg_data[f'eyrewell-irrigated_pg_m{m:02d}'] / month_len[m] for m in all_months]
    pg2 = [pg_data[f'eyrewell-store600_pg_m{m:02d}'] / month_len[m] for m in all_months]
    inpg = np.array([pg1, pg2]).transpose()

    ndays = np.array([month_len[m] for m in all_months])

    model = SimpleDairyModel(all_months,
                             istate=[0, 0], pg=inpg, ifeed=[0, 0], imoney=[0, 0],
                             sup_feed_cost=sup_cost, product_price=product_price, monthly_input=True)
    model.run_model(printi=False)
    fig, axs = model.plot_results('lactating_cow_fraction', 'dry_cow_fraction', 'replacement_fraction',
                                  mult_as_lines=True)
    fig.tight_layout()
    fig, axs = model.plot_results('state', 'feed', 'money')
    fig.tight_layout()
    fig, axs = model.plot_results('cum_feed_import', 'feed')
    fig.tight_layout()
    fig, axs = model.plot_results('cum_feed_import', 'feed_scarcity_cost', 'feed_cost')
    fig.tight_layout()
    fig, axs = model.plot_results('feed_demand', 'home_growth_me')
    fig.tight_layout()
    fig, axs = model.plot_results('surplus_homegrown', 'sup_feed_needed', )
    fig.tight_layout()
    fig, axs = model.plot_results('marginal_cost', 'marginal_benefit', marker='o')
    fig.tight_layout()
    plt.show()


def test_average_year_scarcity():
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = 8.09

    print(f'sup_cost (kgms): {sup_cost * 123.19}, product_price: {product_price}')

    data_path = Path(__file__).parent.joinpath('test_data', 'baseline-mod-median.csv')
    pg_data = pd.read_csv(data_path, index_col=0)['0']
    all_months = np.array([(7 - 1 + i) % 12 + 1 for i in range(12)] * 3)
    pg1 = [pg_data[f'eyrewell-irrigated_pg_m{m:02d}'] / month_len[m] for m in all_months]
    pg2 = [pg_data[f'eyrewell-store600_pg_m{m:02d}'] / month_len[m] for m in all_months]
    inpg = np.array([pg1, pg2]).transpose()
    inpg[np.in1d(all_months, [12, 1, ])] *= 0

    ndays = np.array([month_len[m] for m in all_months])

    model = DairyModelWithScarcity(all_months,
                                   istate=[0, 0], pg=inpg, ifeed=[0, 0], imoney=[0, 0],
                                   sup_feed_cost=sup_cost, product_price=product_price, monthly_input=True)
    model.run_model(printi=False)
    fig, axs = model.plot_results('lactating_cow_fraction', 'dry_cow_fraction', 'replacement_fraction',
                                  mult_as_lines=True)
    fig.tight_layout()
    fig, axs = model.plot_results('state', 'feed', 'money')
    fig.tight_layout()
    fig, axs = model.plot_results('cum_feed_import', 'feed')
    fig.tight_layout()
    fig, axs = model.plot_results('cum_feed_import', 'feed_scarcity_cost', 'feed_cost')
    fig.tight_layout()
    fig, axs = model.plot_results('feed_demand', 'home_growth_me')
    fig.tight_layout()
    fig, axs = model.plot_results('surplus_homegrown', 'sup_feed_needed', )
    fig.tight_layout()
    fig, axs = model.plot_results('marginal_cost', 'marginal_benefit', marker='o')
    fig.tight_layout()
    plt.show()


def test_varying_payout():
    outdir = Path.home().joinpath('Downloads', 'simple_dairy_model_varying_payout')
    outdir.mkdir(exist_ok=True)
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = np.arange(1, 9, 1)
    sims = {i: f'payout ${p}' for i, p in enumerate(product_price)}
    nsims = len(product_price)

    print(f'sup_cost (kgms): {sup_cost * 123.19}, product_price: {product_price}')

    data_path = Path(__file__).parent.joinpath('test_data', 'baseline-mod-median.csv')
    pg_data = pd.read_csv(data_path, index_col=0)['0']
    all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)]
    pg1 = [pg_data[f'eyrewell-irrigated_pg_m{m:02d}'] / month_len[m] for m in all_months]
    inpg = np.array([pg1] * nsims).transpose()

    ndays = np.array([month_len[m] for m in all_months])

    model = SimpleDairyModel(all_months,
                             istate=[0] * nsims, pg=inpg, ifeed=[0] * nsims, imoney=[0] * nsims,
                             sup_feed_cost=sup_cost,
                             product_price=np.repeat(product_price[np.newaxis], len(all_months), axis=0),
                             monthly_input=True)
    model.run_model(printi=False)
    fig, axs = model.plot_results('lactating_cow_fraction', 'dry_cow_fraction', 'replacement_fraction',
                                  mult_as_lines=True, sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('cow_fractions.png'))
    fig, axs = model.plot_results('cum_feed_import', 'feed', 'money', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('feed_money.png'))
    fig, axs = model.plot_results('feed_demand', 'home_growth_me', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('feed_demand.png'))
    fig, axs = model.plot_results('surplus_homegrown', 'sup_feed_needed', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('surplus_homegrown.png'))
    fig, axs = model.plot_results('marginal_cost', 'marginal_benefit', marker='o')
    fig.tight_layout()
    fig.savefig(outdir.joinpath('marginal_cost_benefit.png'))
    plt.show()


def test_varying_sup_cost():
    outdir = Path.home().joinpath('Downloads', 'simple_dairy_model_varying_sup_cost')
    outdir.mkdir(exist_ok=True)
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = np.array([400, 1000, 1500, 2000, 4000, 6000, 8000, 10000])
    product_price = 8.09
    sims = {i: f'sup ${p}/t' for i, p in enumerate(sup_cost)}
    nsims = len(sup_cost)
    sup_cost = sup_cost / 1000 / mj_per_kg_dm

    print(f'sup_cost (kgms): {sup_cost * 123.19}, product_price: {product_price}')

    data_path = Path(__file__).parent.joinpath('test_data', 'baseline-mod-median.csv')
    pg_data = pd.read_csv(data_path, index_col=0)['0']
    all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
    pg1 = [pg_data[f'eyrewell-irrigated_pg_m{m:02d}'] / month_len[m] for m in all_months]
    inpg = np.array([pg1] * nsims).transpose()

    ndays = np.array([month_len[m] for m in all_months])

    model = SimpleDairyModel(all_months,
                             istate=[0] * nsims, pg=inpg, ifeed=[0] * nsims, imoney=[0] * nsims,
                             sup_feed_cost=np.repeat(sup_cost[np.newaxis], len(all_months), axis=0),
                             product_price=product_price,
                             monthly_input=True)
    model.run_model(printi=False)
    fig, axs = model.plot_results('lactating_cow_fraction', 'dry_cow_fraction', 'replacement_fraction',
                                  mult_as_lines=True, sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('cow_fractions.png'))
    fig, axs = model.plot_results('cum_feed_import', 'feed', 'money', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('feed_money.png'))
    fig, axs = model.plot_results('feed_demand', 'home_growth_me', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('feed_demand.png'))
    fig, axs = model.plot_results('surplus_homegrown', 'sup_feed_needed', sims=sims)
    fig.tight_layout()
    fig.savefig(outdir.joinpath('surplus_homegrown.png'))
    fig, axs = model.plot_results('marginal_cost', 'marginal_benefit', marker='o')
    fig.tight_layout()
    fig.savefig(outdir.joinpath('marginal_cost_benefit.png'))
    plt.show()


def test_historical_data(): # todo bayesian optimisation for parameters? --> might be the simplest way...
    from komanawa.slmacc_csra import get_historical_pg_data

    history = get_historical_pg_data('eyrewell', 'irrigated')
    history['water_year'] = history['year'] + (history['month'] >= 7).astype(int)
    history = history.groupby(['water_year', 'month']).mean()
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = 8.09
    all_months = np.array([(7 - 1 + i) % 12 + 1 for i in range(12)])
    inpg = np.array([history.loc[wy].loc[all_months, 'pg'].values for wy in history.index.levels[0]]).transpose()
    # todo start here...
    pass

    # todo pull out the percent of feed import from the historical data

    raise NotImplementedError


if __name__ == '__main__':
    test_historical_data()
    # test_average_year_scarcity()
    # test_average_year()
    # test_varying_payout()
    # test_varying_sup_cost()
