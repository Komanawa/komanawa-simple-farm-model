"""
created matt_dumont 
on: 6/27/24
"""

# todo eventually add this to test_simple_dairy_model.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from komanawa.simple_farm_model.simple_dairy_model import SimpleDairyModel, month_len


def test_average_year():
    # todo slightly concernted with the milk production feed tie here...
    # todo once a day milking never seems to be worth it..., stops any other mitigations.  essentially tied to
    # todo at this supplment cost it is always worth bringing in more feed, even if it is expensive.
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
                             istate=[0, 0], pg=inpg, ifeed=[0,0], imoney=[0, 0],
                             sup_feed_cost=sup_cost, product_price=product_price, monthly_input=True)
    model.run_model(printi=False)
    fig, axs = model.plot_results('lactating_cow_fraction', 'dry_cow_fraction', 'replacement_fraction',
                                  mult_as_lines=True)
    fig.tight_layout()
    fig, axs = model.plot_results('state', 'feed', 'money')
    fig.tight_layout()
    fig, axs = model.plot_results('cum_feed_import', 'feed')
    fig.tight_layout()
    fig, axs = model.plot_results('feed_demand', 'home_growth_me')
    fig.tight_layout()
    fig, axs = model.plot_results('surplus_homegrown', 'sup_feed_needed', )
    fig.tight_layout()
    fig, axs = model.plot_results('marginal_cost', 'marginal_benefit', marker='o')
    fig.tight_layout()
    plt.show()


def test_historical_data():
    raise NotImplementedError


if __name__ == '__main__':
    test_average_year()
