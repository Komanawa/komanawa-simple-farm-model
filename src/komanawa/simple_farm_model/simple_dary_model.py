"""
created matt_dumont 
on: 3/07/23
"""
import numpy as np
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel
import pandas as pd

class SimpleDairyFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }

    month_reset = 8  # trigger farm reset on day 1 in August
    default_feed_cost = 0.40  # $/kgDM
    default_prod_value = 7.13  # $/kgMS

    base_cow = 3.5
    state_cow_modifier = {  # i value: (nmiking, stock levels)
        1: 0.95,  # ('2aday', 'low'),
        2: 1.00,  # ('2aday', 'norm'),
        3: 0.95,  # ('1aday', 'low'),
        4: 1.00,  # ('1aday', 'norm'),
    }
    seasonal_cow_modifer = {
        8: 0.90,
        9: 0.95,
        10: 1.00,
        11: 1.00,
        12: 0.95,
        1: 0.95,
        2: 0.90,
        3: 0.85,
        4: 0.80,
        5: 0.80,
        6: 0.00,
        7: 0.00,
    }

    # calculate feed/ cows kgDM/cow
    feed_per_cow_state_modifer = {  # i value: (nmiking, stock levels)
        1: 1.,  # ('2aday', 'low'),
        2: 1.,  # ('2aday', 'norm'),
        3: .90,  # ('1aday', 'low'),
        4: .90,  # ('1aday', 'norm'),
    }

    feed_per_cow_monthly = {
        # excludes expected typical supplimnations which are included in base costs:
        # todo I need to fix the above, bad assumption
        8: 13.855,
        9: 14.67,
        10: 14.67,
        11: 16.468,
        12: 17.2735,
        1: 16.625,
        2: 16.8,
        3: 15.39,
        4: 14.904,
        5: 14.904,
        6: 0.00,
        7: 0.00,

    }

    production_per_cow = {
        8: 1.5,
        9: 1.8,
        10: 1.9,
        11: 1.9,
        12: 2.1,
        1: 2.0,
        2: 1.8,
        3: 1.5,
        4: 1.4,
        5: 1.2,
        6: 0.0,
        7: 0.0,

    }
    production_state_modifier = {  # i value: (nmiking, stock levels)
        1: 1.0,  # ('2aday', 'low'),
        2: 1.0,  # ('2aday', 'norm'),
        3: 0.78,  # ('1aday', 'low'),
        4: 0.78,  # ('1aday', 'norm'),
    }

    base_running_cost = 4.11 * 469 * 3.5 / 365  # $/ha
    state_running_cost_modifier = {  # i value: (nmiking, stock levels)
        1: 0.99,  # ('2aday', 'low'),
        2: 1.00,  # ('2aday', 'norm'),
        3: 0.98,  # ('1aday', 'low'),
        4: 0.99,  # ('1aday', 'norm'),
    }

    debt_servicing = 34968.32421875  # $/ha/yr   # todo confirm units, NOPE THIS IS DEBT LEVELS, how are we managing debt servicing?
# todo need 3 buckets on pasture , on farm sup, external,  PAUL NEEDS TO TAKE OU
#  cost to move from pasture to farm sup,
#  no cost to use on farm sup
    def calculate_feed_needed(self, i_month, month, current_state):
        # todo
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'

        # calculate number of cows / ha
        state_cow_modifier = np.array(self.state_cow_modifier.values())
        ncows = self.base_cow * self.seasonal_cow_modifer[month] * state_cow_modifier[current_state - 1]

        feed_per_cow_state_modifer = np.array(self.feed_per_cow_state_modifer.values())
        feed_per_cow = self.feed_per_cow_monthly[month] * feed_per_cow_state_modifer[current_state - 1]

        out = ncows * feed_per_cow  # kgDM/ha

        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_production(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        # calculate number of cows / ha
        state_cow_modifier = np.array(self.state_cow_modifier.values())
        ncows = self.base_cow * self.seasonal_cow_modifer[month] * state_cow_modifier[current_state - 1]

        # production per cow
        prod_per_cow = np.array(self.feed_per_cow_state_modifer.values())
        prod_per_cow = self.production_per_cow[month] * prod_per_cow[current_state - 1]

        out = ncows * prod_per_cow  # kg MS/ha
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_sup_feed(self, i_month, month, current_state):  # todo need feedback from Paul on trigger levels
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')  # todo

    def calculate_running_cost(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'

        running_cost = self.base_running_cost * np.array(self.state_running_cost_modifier.values())[current_state - 1]
        idx = self.model_feed_imported[i_month] > 0
        running_cost[idx] *= 1.01  # running cost increase by 1% if feed is imported
        out = running_cost
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out

    def calculate_next_state(self, i_month, month, current_state): # todo need Paul's feedback on this (for tigger levels)
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')  # todo

    def reset_state(self, i_month):
        out = np.zeros(self.model_shape[1]) + 2  # reset to state 2: 2 a day milking and normal stocking
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        return out
