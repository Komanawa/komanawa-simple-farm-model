"""
created matt_dumont 
on: 4/07/23
"""
import matplotlib.pyplot as plt
# todo refactor to proper tests..., test more things...
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel
import numpy as np
import pandas as pd
from pathlib import Path


class DummySimpleFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }
    month_reset = 7  # trigger farm reset on day 1 in July
    homegrown_efficiency = 1
    supplemental_efficiency = 1
    sup_feedout_cost = 0.4
    homegrown_store_efficiency = 1
    homegrown_storage_cost = 0.1

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 8
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_production(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1]) + 2
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def reset_state(self, i_month, ):
        out = np.zeros(self.model_shape[1]) + 2
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def convert_pg_to_me(self, pg, current_state):
        return pg * 1

    def import_feed_and_change_state(self, i_month, month, current_state, current_feed):
        """
        import feed and change state

        :param i_month: month index
        :param month: month
        :param current_state: current state
        :param current_feed: current feed
        :return:
        """
        assert pd.api.types.is_integer(i_month), f'month must be int, got {type(i_month)}'
        idx = current_feed < 100
        feed_imported = np.zeros(self.nsims)
        feed_imported[idx] = 1000 - current_feed[idx]
        sup_feed_cost = feed_imported * self.sup_feed_cost[i_month]
        next_state = np.zeros(self.nsims) + 1
        return next_state, feed_imported, sup_feed_cost


def test_basic_farm_model():
    all_months = list(range(1, 13)) + list(range(1, 13)) + list(range(1, 13))

    farm = DummySimpleFarm(all_months, istate=np.ones(5), pg=all_months, ifeed=np.arange(5) * 200,
                           imoney=np.arange(5) * 200, sup_feed_cost=0.4, product_price=3.5,
                           monthly_input=True)
    print('farm model shape=', farm.model_shape)
    farm.run_model()
    farm.plot_results('state', 'feed', 'money')
    farm.plot_results('state', 'feed', 'money', sims=None, mult_as_lines=False, twin_axs=False)
    farm.plot_results('feed', 'money', sims=None, mult_as_lines=True, twin_axs=True)
    farm.plot_results('feed', 'money', sims=[1, 2], mult_as_lines=True, twin_axs=True)

    # test different x in plots, check
    farm.plot_results('feed', x='money', sims=None, mult_as_lines=True, twin_axs=True)

    # test different x in non line plots
    farm.plot_results('feed', x='money', sims=None, mult_as_lines=False, twin_axs=False)

    # test save to nc
    farm.save_results_nc(Path.home().joinpath('Downloads/testfarm_outputs.nc'))

    # test save to csv
    farm.save_results_csv(Path.home().joinpath('Downloads/testfarm_outputs'))

    # test load from nc
    farm2 = DummySimpleFarm.from_input_file(Path.home().joinpath('Downloads/testfarm_outputs.nc'))

    # confirm that farm and farm2 are the same
    bad = []
    for k, v in farm.__dict__.items():
        if k in []:
            continue
        if isinstance(v, np.ndarray):
            if not np.all(np.isclose(v, getattr(farm2, k), equal_nan=True)):
                bad.append(k)
        else:
            if v != getattr(farm2, k):
                bad.append(k)
    assert len(bad) == 0, f'farm and farm2 are not the same, bad keys={bad}'

    plt.show()
    pass


if __name__ == '__main__':
    test_basic_farm_model()
