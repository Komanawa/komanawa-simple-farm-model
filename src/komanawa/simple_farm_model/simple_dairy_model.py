"""
created: matt_dumont
on: 6/15/24

This module contains the SimpleDairyModel class which is a subclass of the BaseSimpleFarmModel class. The SimpleDairyModel class is used to simulate a simple dairy farm model. It includes methods for calculating feed requirements, production, and marginal costs/benefits, among other things.

The General process is defined in the BaseSimpleFarmModel class.  There are a few new features in the SimpleDairyModel class that are not in the BaseSimpleFarmModel class.  These include:

Cow Buckets:
=============================

The model contains 3 distinct cow buckets, which are parameterised as fraction of peak lactacting cows per bucket. These buckets are:

* Lactating_cow_fraction
* Dry_cow_fraction
* Replacement_fraction

Note that Lactating_cow_fraction and dry_cow_fraction buckets ranges between 0-1 and the sum of both of these buckets must be <= 1, which means we can destock as well as move cows (or partial cows) between buckets.

The replacements are set by the replacement rate (default 22%).  Additional replacements are added at the end of November (to 44%) and replacements are transitioned into dry cows at the end of April.  All cows are dried off in May, June, and July.

We’re working on a pre-ha basis, but I’m assuming that the farm is significantly large enough that rounding to the nearest 0.001 is sensible.  I have made it, so we can easily change this precision.

Cows in each bucket have different feed requirements, and the model calculates the feed requirements for each bucket separately.

Cow loss
--------------------

To simulate typical die off rates for cows, we apply a pro-rata loss of cows per day.  This is applied on the last day of each month.  The loss rates are defined as:

* lactating_cow_loss: 2.1% loss per year
* replacement_cow_loss: simplifying assumption no loss per year.
* dry_cow_loss: 2.1% loss per year

These loss rates are set in the class:

* lactating_cow_loss
* replacement_cow_loss
* dry_cow_loss

Feed import and state change
=============================

State changes are fundamentally defined by a low feed store and a comparison between the maringal cost/benefit of the state change.

How to trigger state change
-------------------------------

The model assess the marginal cost/benefit of a state change on the 1st and 15th day of each month this includes the amount of feed on hand, but excludes the cost of feeding out stored feed. If the marginal cost is less than the marginal benefit, the state change is implemented.

State change quantization and progression
--------------------------------------------

The fundamental states are defined as:

* 0: 2 a day milking
* 1: 1 a day milking

The model allows for shifting of cows between buckets (or reduction in stock), but these changes are optimised at each step change based on the assumed future product price, pasture growth, and supplementary feed cost.

Feed import process.
-----------------------

There is also a feed import for x(13%) of the total feed required per year.  This is done at the start of each year (July 1).

For additional feed imports, rather than having a just in time feed import process my plan is to import feed whenever the store falls below some threshold (in MJ). Then when we import feed we import N(n=7) days of supplemental feed (that is the amount of supplemental feed imported over the last n days)


Marginal cost/benefit calculation
===================================

The marginal cost and benefit of a potential state change is assessed on the 1st and 15th day of each month.  T

The marginal benefit is calculated as:

* The cost of the feed required to maintain the current state through the rest of the year (to reset) (including feed out cost)
* vs the cost of the feed required to maintain the alternate state through the rest of the year (to reset) (including feed out cost)

The marginal cost is calculated as:

* the income from the current state through the rest of the year (to reset)
* the income from the alternate state through the rest of the year (to reset)

In this calculation there are several simplifying assumptions:

1. all home grown feed is utilized (excludes storage costs)
2. supplemental feed is always available and will be purchased in for the rest of the year (to reset)

These calculations require a future projection of pasture growth, supplemental feed costs, and product prices.

The model allows for three different ways to set these values:

1. use the True values.  all of these variables are prescribed in the init, therefore the model can run the analysis on the true values.  This is accomplished by setting the `marginal_benefit_from_true` and `marginal_cost_from_true` to True.  This is the default setting.
2. use the monthly mean values from the init (this is the default setting if `marginal_benefit_from_true` and `marginal_cost_from_true` are false).  Here the monthly mean value for the full simulation is calculated from the sim (e.g. each of nsims may have different monthly mean values).
3. user specified.  Alternatively the user can specify the mean values to use for the marginal cost/benefit calculations. This allows the user to create `optimistic` and `pessimistic` farmer scenarios.  This is done with the `set_mean_pg`, `set_mean_sup_cost`, and `set_mean_product_price` methods.


State change options
=======================

The model allows for 4 different state changes, only one of which can be implemented in each fortnightly period.  These are:

* 0: no change
* 1: cull cows
* 2: dryoff cows
* 3: go to once a day milking

The model assesses the alternate step change in the following order:

1.  Before end of February (months 8-2) go to:
    1. once-a-day milking
    2. this cull up to the replacement rate (no on-going feed requirement)
    3. Post this dry-off (significantly reduced feed demand).
2. After end of Feb, but before end of April (months 3-4) go to:
    1. Cull up to the replacement rate
    2. once-a-day milking
    3. dry-off stock
3. For months 5, 6, 7, all cows are dried off regardless so
    1. No change
"""
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel, month_len
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from copy import deepcopy
from pathlib import Path

mj_per_kg_dm = 11  # MJ ME /kg DM
monly_ms_prod = {  # kgMS per lactating cow per day
    1: 57.47,
    2: 49.10,
    3: 50.13,
    4: 45.89,
    5: 0,  # based on discussions with andrew curtis, not worth including may prod. prev.45.42,
    6: 0,  # yep
    7: 0,  # not really much lactation in July, so we'll assume 0
    8: 56.92,
    9: 64.27,
    10: 68.17,
    11: 61.57,
    12: 60.67
}
daily_ms_prod = {m: v / month_len[m] for m, v in monly_ms_prod.items()}

dry_cow_feed = {  # kgDM per cow per day
    7: 9.3,
    8: 6,
    9: 6,
    10: 6,
    11: 6,
    12: 6,
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 6.8,
    6: 8.3,
}
dry_cow_feed = {k: v * mj_per_kg_dm for k, v in dry_cow_feed.items()}
replacement_feed = {m: 6.4 * mj_per_kg_dm for m in range(1, 13)}
lactating_feed = {k: v * 123.19 for k, v in daily_ms_prod.items()}  # assume 123.19 MJ ME/kg DM


class SimpleDairyModel(BaseSimpleFarmModel):
    """
    A simple dairy farm model that simulates a dairy farm. The model includes methods for calculating feed requirements, production, and marginal costs/benefits, among other things.
    """
    precision = 3  # round to the nearest 0.001 cows
    states = {
        0: True,  # 2 a day, miliking
        1: False  # 1 a day milking
    }
    mean_sup_cost = None  # set internally
    mean_pg = None  # set internally
    annual_supplement_import = 0.14  # 13% of total feed required per year (less any stored feed)
    annual_supplement_import= 0.0  # 0% of total feed required per year (less any stored feed), # todo seeing if this gives more flexibility

    ndays_feed_import = 7  # when needing to import feed, import 7 days worth of feed (from 7 days supplemental needed)
    state_change_days = [1, 15]  # allow for a change in farm state on day 1 and 15 of each month
    month_reset = 7  # trigger farm reset on day 1 in August,
    homegrown_efficiency = 0.8
    supplemental_efficiency = 0.9
    sup_feedout_cost = 120 / 1000 / mj_per_kg_dm  # 120/ton DM --> convert to ME, MJ
    homegrown_store_efficiency = 1
    homegrown_storage_cost = 175 / 1000 / mj_per_kg_dm  # 175/ton/DM --> convert to ME, MJ
    one_a_daymilk_production_fraction = 1 - 0.13  # 13% less milk production on 1-a-day
    _start_lactating_cow_fraction = 0  # % of cows are lactating on day 1  (july)#
    _start_dry_cow_fraction = 1  # 100% of cows are dry on day 1 (july)
    _start_replacement_fraction = None  # set internally to replacement rate
    replacement_rate = 0.22  # 22% of cows are replacements
    peak_lact_cow_per_ha = (3.48 + 2.82)/2/1.44  # (dairy platform density + replacement density) / 2 / 1.44
    kgms_per_lac_cow = deepcopy(daily_ms_prod)  # kgMS per lactating cow per day
    feed_per_cow_lactating = deepcopy(lactating_feed)
    feed_per_cow_dry = deepcopy(dry_cow_feed)
    feed_per_cow_replacement = deepcopy(replacement_feed)
    feed_fraction_1aday = 0.87  # 13% less feed required on 1-a-day
    feed_store_trigger = None  # set internally to two weeks of average feed requirements
    marginal_benefit_from_true = True  # calculate marginal benefit from true prices, if False, use average price
    marginal_cost_from_true = True  # calculate marginal benefit from true prices, if False, use average price
    lactating_cow_loss = 0.021  # 2.1% loss per year
    replacement_cow_loss = 0  # simplifying assumption, no loss of replacements
    dry_cow_loss = 0.021  # 2.1% loss per year
    allow_1aday = False  # allow for 1 a day milking, abandoning... drying off some small percentage of cows also accomplishes the same thing given we don't have any running costs.

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price, monthly_input=True):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number float or np.ndarray shape (nsims,)
        :param imoney: initial money number float or np.ndarray shape (nsims,)
        :param sup_feed_cost: cost of supplementary feed $/MJ float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param product_price: income price $/kg product float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param monthly_input: if True, monthly input, if False, daily input (365 days per year)
        """
        if self.allow_1aday:
            self.calc_next_state_quant = self.calc_next_state_quant_1aday
        else:
            self.calc_next_state_quant = self.calc_next_state_quant_not_1aday

        assert self.replacement_cow_loss == 0, 'replacement cow loss must be 0 as we are not simulating replacement cow loss'
        assert self.lactating_cow_loss == self.dry_cow_loss, 'lactating and dry cow loss must be the same'
        self._start_dry_cow_fraction = self._start_dry_cow_fraction + self.lactating_cow_loss / 12
        self._start_replacement_fraction = self.replacement_rate
        # set expected minimum of dry/lactating cow fraction
        self.min_cow = {  # the maximum cull level for each month
            5: 1 + self.dry_cow_loss / 12 * 2,
            6: 1 + self.dry_cow_loss / 12 * 1,
            7: 1,
            8: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 12,
            9: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 11,
            10: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 10,
            11: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 9,
            12: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 8,
            1: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 7,
            2: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 6,
            3: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 5,
            4: 1 - self.replacement_rate + self.dry_cow_loss / 12 * 4,
        }

        self.max_cow = {
            5: 1 + self.dry_cow_loss / 12 * 3,
            6: 1 + self.dry_cow_loss / 12 * 2,
            7: 1 + self.dry_cow_loss / 12,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
        }
        # add more input/outputs

        super().__init__(all_months=all_months, istate=istate, pg=pg,
                         ifeed=ifeed, imoney=imoney, sup_feed_cost=sup_feed_cost, product_price=product_price,
                         monthly_input=monthly_input)

        # call reset state to pull the initial annual feed import.
        self.reset_state(0, self.model_feed[0], self.model_money[0])

        # identify which states are 1 a day milking
        self.milk_1_aday_states = [i for i, a_day_2 in self.states.items() if not a_day_2]

        self.set_mean_pg(pg_dict=None, pg_raw=self.pg, months=self.all_months)
        self.set_mean_sup_cost(sup_dict=None, sup_cost_raw=self.sup_feed_cost, months=self.all_months)
        self.set_mean_product_price(prod_price_dict=None, prod_price_raw=self.product_price, months=self.all_months)

        # set feed store trigger to 2 weeks of fully lactating feed requirements, with 2 * replacements
        start_cows = self._start_lactating_cow_fraction + self._start_dry_cow_fraction
        max_lact_feed = max(self.feed_per_cow_lactating.values())
        max_rep_feed = max(self.feed_per_cow_replacement.values())

        self.feed_store_trigger = 1000

    def _update_object_dict(self):
        super()._update_object_dict()
        self.obj_dict['lactating_cow_fraction'] = 'out_lactating_cow_fraction'
        self.obj_dict['dry_cow_fraction'] = 'out_dry_cow_fraction'
        self.obj_dict['replacement_fraction'] = 'out_replacement_fraction'
        self.obj_dict['feed_lactating'] = 'out_feed_lactating'
        self.obj_dict['feed_dry'] = 'out_feed_dry'
        self.obj_dict['feed_replacement'] = 'out_feed_replacement'
        self.obj_dict['cur_fut_prod'] = 'out_cur_fut_prod'
        self.obj_dict['alt_fut_prod'] = 'out_alt_fut_prod'
        self.obj_dict['marginal_cost'] = 'out_marginal_cost'
        self.obj_dict['alt_fut_feed_def'] = 'out_alt_fut_feed_def'
        self.obj_dict['cur_fut_feed_def'] = 'out_cur_fut_feed_def'
        self.obj_dict['marginal_benefit'] = 'out_marginal_benefit'

        self.long_name_dict['lactating_cow_fraction'] = 'Lactating Cow Fraction (fraction, 0-1)'
        self.long_name_dict['dry_cow_fraction'] = 'Dry Cow Fraction (fraction, 0-1)'
        self.long_name_dict['replacement_fraction'] = 'Replacement Cow Fraction (fraction, 0-1)'
        self.long_name_dict['feed_lactating'] = 'Feed Lactating required on a given day (MJ)'
        self.long_name_dict['feed_dry'] = 'Feed Dry required on a given day (MJ)'
        self.long_name_dict['feed_replacement'] = 'Feed Replacement required on a given day (MJ)'
        self.long_name_dict['cur_fut_prod'] = 'Current Future Scenario Production for MC/MB ($)'
        self.long_name_dict['alt_fut_prod'] = 'Alternate Future Scenario Production for MC/MB ($)'
        self.long_name_dict['marginal_cost'] = 'Marginal Cost for MC/MB ($)'
        self.long_name_dict['alt_fut_feed_def'] = 'Alternate Future Feed Deficit for MC/MB (MJ)'
        self.long_name_dict['cur_fut_feed_def'] = 'Current Future Feed Deficit for MC/MB (MJ)'
        self.long_name_dict['marginal_benefit'] = 'Marginal Benefit for MC/MB ($)'

    def _set_arrays(self, all_months, istate, ifeed, imoney, pg, sup_feed_cost, product_price, monthly_input):
        super()._set_arrays(all_months, istate, ifeed, imoney, pg, sup_feed_cost, product_price, monthly_input)

        # make cow buckets...
        self.lactating_cow_fraction = np.zeros(self.nsims) + self._start_lactating_cow_fraction
        self.dry_cow_fraction = np.zeros(self.nsims) + self._start_dry_cow_fraction
        self.replacement_fraction = np.zeros(self.nsims) + self._start_replacement_fraction

        # make outcow buckets
        self.out_lactating_cow_fraction = np.zeros(self.model_shape)
        self.out_dry_cow_fraction = np.zeros(self.model_shape)
        self.out_replacement_fraction = np.zeros(self.model_shape)
        self.out_lactating_cow_fraction[0] = self.lactating_cow_fraction
        self.out_dry_cow_fraction[0] = self.dry_cow_fraction
        self.out_replacement_fraction[0] = self.replacement_fraction
        self.out_feed_lactating = np.zeros(self.model_shape) * np.nan
        self.out_feed_dry = np.zeros(self.model_shape) * np.nan
        self.out_feed_replacement = np.zeros(self.model_shape) * np.nan
        self.out_cur_fut_prod = np.zeros(self.model_shape) * np.nan
        self.out_alt_fut_prod = np.zeros(self.model_shape) * np.nan
        self.out_marginal_cost = np.zeros(self.model_shape) * np.nan
        self.out_alt_fut_feed_def = np.zeros(self.model_shape) * np.nan
        self.out_cur_fut_feed_def = np.zeros(self.model_shape) * np.nan
        self.out_marginal_benefit = np.zeros(self.model_shape) * np.nan

    def set_mean_product_price(self, prod_price_dict, prod_price_raw=None, months=None):
        """
        This method can be used to set a bespoke mean pasture growth for each month to calculate marginal cost/benefit by default this is called on self.prod_price in the init method

        one of prod_price_dict or prod_price_raw and months must be set

        :param prod_price_dict: None or dict of pasture growth values for each month, keys are 1-12, values are float or np.ndarray shape (nsims,)
        :param prod_price_raw: None or np.ndarray shape (mon_len, nsims) pasture growth values for each day
        :param months: None or np.ndarray shape (nsims,) of month values for each day
        :return:
        """
        assert (prod_price_dict is None and (prod_price_raw is not None and months is not None)
                or (prod_price_dict is not None and prod_price_raw is None and months is None)), (
            'prod_price_dict or prod_price_raw and months must be set')

        if prod_price_dict is not None:
            for k, v in prod_price_dict.items():
                if pd.api.types.is_number(v):
                    prod_price_dict[k] = np.zeros(self.nsims) + v
                else:
                    v = np.atleast_1d(v)
                    assert v.shape == (self.nsims,), f'prod_price values must be shape {self.nsims=}, got {v.shape}'
                    prod_price_dict[k] = v
        else:
            assert isinstance(prod_price_raw, np.ndarray) and isinstance(months, np.ndarray)
            assert prod_price_raw.shape == self.model_shape, f'prod_price_raw must be shape {self.model_shape}, got {prod_price_raw.shape}'
            assert months.shape == (
                self.model_shape[0],), f'months must be shape {self.model_shape[:1]=}, got {months.shape}'
            prod_price_dict = {}
            for m in range(1, 13):
                idx = months == m
                prod_price_dict[m] = prod_price_raw[idx].mean(axis=0)
        assert isinstance(prod_price_dict, dict), f'prod_price must be dict, got {type(prod_price_dict)}'
        assert set(prod_price_dict.keys()) == set(
            range(1, 13)), f'prod_price keys must be 1-12, got {prod_price_dict.keys()}'
        self.mean_prod_price = prod_price_dict

    def set_mean_pg(self, pg_dict, pg_raw=None, months=None):
        """
        This method can be used to set a bespoke mean pasture growth for each month to calculate marginal cost/benefit by default this is called on self.pg in the init method

        one of pg_dict or pg_raw and months must be set

        :param pg_dict: None or dict of pasture growth values for each month, keys are 1-12, values are float or np.ndarray shape (nsims,)
        :param pg_raw: None or np.ndarray shape (mon_len, nsims) pasture growth values for each day
        :param months: None or np.ndarray shape (nsims,) of month values for each day
        :return:
        """
        assert (pg_dict is None and (pg_raw is not None and months is not None)
                or (pg_dict is not None and pg_raw is None and months is None)), (
            'pg_dict or pg_raw and months must be set')

        if pg_dict is not None:
            for k, v in pg_dict.items():
                if pd.api.types.is_number(v):
                    pg_dict[k] = np.zeros(self.nsims) + v
                else:
                    v = np.atleast_1d(v)
                    assert v.shape == (self.nsims,), f'pg values must be shape {self.nsims=}, got {v.shape}'
                    pg_dict[k] = v
        else:
            assert isinstance(pg_raw, np.ndarray) and isinstance(months, np.ndarray)
            assert pg_raw.shape == self.model_shape, f'pg_raw must be shape {self.model_shape}, got {pg_raw.shape}'
            assert months.shape == (
                self.model_shape[0],), f'months must be shape {self.model_shape[:1]=}, got {months.shape}'
            pg_dict = {}
            for m in range(1, 13):
                idx = months == m
                pg_dict[m] = pg_raw[idx].mean(axis=0)

        assert isinstance(pg_dict, dict), f'pg must be dict, got {type(pg_dict)}'
        assert set(pg_dict.keys()) == set(range(1, 13)), f'pg keys must be 1-12, got {pg_dict.keys()}'
        self.mean_pg = pg_dict

    def set_mean_sup_cost(self, sup_dict, sup_cost_raw=None, months=None):
        """
        This method can be used to set a bespoke mean supplementary feed cost for each month to calculate marginal cost/benefit by default this is called on self.sup_feedout_cost in the init method

        one of sup_dict or sup_cost_raw and months must be set

        :param sup_dict: None or dict of supplementary feed cost values for each month, keys are 1-12, values are float or np.ndarray shape (nsims,)
        :param sup_cost_raw: None or np.ndarray shape (mon_len, nsims) supplementary feed cost values for each day
        :param months: None or np.ndarray shape (nsims,) of month values for each day
        :return:
        """
        assert (sup_dict is None and (sup_cost_raw is not None and months is not None)
                or (sup_dict is not None and sup_cost_raw is None and months is None)), (
            'sup_dict or sup_cost_raw and months must be set')

        if sup_dict is not None:
            for k, v in sup_dict.items():
                if pd.api.types.is_number(v):
                    sup_dict[k] = np.zeros(self.nsims) + v
                else:
                    v = np.atleast_1d(v)
                    assert v.shape == (self.nsims,), f'pg values must be shape {self.nsims=}, got {v.shape}'
                    sup_dict[k] = v
        else:
            assert isinstance(sup_cost_raw, np.ndarray) and isinstance(months, np.ndarray)
            assert sup_cost_raw.shape == self.model_shape, f'sup_cost_raw must be shape {self.model_shape}, got {sup_cost_raw.shape}'
            assert months.shape == (
                self.model_shape[0],), f'months must be shape {self.model_shape[:1]=}, got {months.shape}'
            sup_dict = {}
            for m in range(1, 13):
                idx = months == m
                sup_dict[m] = sup_cost_raw[idx].mean(axis=0)
        assert isinstance(sup_dict, dict), f'sup must be dict, got {type(sup_dict)}'
        assert set(sup_dict.keys()) == set(range(1, 13)), f'sup keys must be 1-12, got {sup_dict.keys()}'
        self.mean_sup_cost = sup_dict

    def convert_pg_to_me(self, pg, current_state):
        assert isinstance(pg, np.ndarray), f'pg must be np.ndarray, got {type(pg)}'
        return mj_per_kg_dm * pg

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        idxs = np.in1d(current_state, self.milk_1_aday_states)
        feed_lactating = self.feed_per_cow_lactating[month] * self.lactating_cow_fraction * self.peak_lact_cow_per_ha
        feed_lactating[idxs] *= self.feed_fraction_1aday
        self.out_feed_lactating[i_month] = feed_lactating

        feed_dry = self.feed_per_cow_dry[month] * self.dry_cow_fraction * self.peak_lact_cow_per_ha
        self.out_feed_dry[i_month] = feed_dry

        feed_replacement = self.feed_per_cow_replacement[month] * self.replacement_fraction * self.peak_lact_cow_per_ha
        self.out_feed_replacement[i_month] = feed_replacement

        return feed_lactating + feed_dry + feed_replacement

    def calculate_production(self, i_month, month, current_state):
        max_production = (self.kgms_per_lac_cow[month]
                          * (self.peak_lact_cow_per_ha * self.lactating_cow_fraction).round(self.precision))
        idxs = np.in1d(current_state, self.milk_1_aday_states)
        max_production[idxs] *= self.one_a_daymilk_production_fraction
        return max_production

    def reset_state(self, i_month, current_feed, current_money):
        out = np.zeros(self.nsims)
        assert out.shape == (self.nsims,), f'out must be shape {self.nsims=}, got {out.shape}'

        # annual supplement import
        remaining_months = np.concatenate([np.repeat(m, month_len[m])
                                           for m in [(self.month_reset - 1 + i) % 12 + 1 for i in range(12)]])
        future_product, supplement_cost, annual_feed = self._calc_cost_production(
            remaining_months,
            lact_fraction=np.ones(self.nsims),
            dry_fraction=np.zeros(self.nsims),
            once_aday_idx=np.zeros(self.nsims).astype(bool),
            use_prod_price=np.zeros((len(remaining_months), self.nsims)),
            use_pg=np.zeros((len(remaining_months), self.nsims)),
            current_state=np.zeros(self.nsims),
            current_feed=np.zeros(self.nsims),
            use_feed_cost=np.zeros((len(remaining_months), self.nsims)),
            nsims=self.nsims)
        import_feed = annual_feed * self.annual_supplement_import - current_feed
        import_feed[import_feed < 0] = 0
        self.model_feed_imported[i_month, :] = np.nansum(np.concatenate((import_feed[np.newaxis],
                                                                         self.model_feed_imported[[i_month], :]),
                                                                        axis=0), axis=0)
        current_feed += import_feed
        current_money -= import_feed * self.sup_feed_cost[i_month]

        # keynote reset cow buckets happens in supplemental_action_last

        return out  # all back to twice a day milking

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

        # import feed where below threshold, import ndays worth of feed
        need_feed = current_feed < self.feed_store_trigger
        past_sup_use = self.sup_feed_needed[max(i_month - self.ndays_feed_import + 1, 1):i_month + 1].sum(axis=0)
        feed_imported = np.zeros(self.nsims)
        feed_imported[need_feed] = past_sup_use[need_feed]
        sup_feed_cost = feed_imported * self.sup_feed_cost[i_month]
        new_state = np.zeros(self.nsims) + current_state

        if (self.all_days[i_month] in self.state_change_days) and (month not in [5, 6, 7]):
            # allow state change
            (marginal_cost, marginal_benefit, next_action, alt_lact_fraction,
             alt_dry_fraction) = self.calc_marginal_cost_benefit(i_month, month, current_state, current_feed)
            idx = marginal_cost < marginal_benefit
            self.dry_cow_fraction[idx] = alt_dry_fraction[idx]
            self.lactating_cow_fraction[idx] = alt_lact_fraction[idx]
            new_once_aday = next_action == 3
            new_state[new_once_aday & idx] = 1

        assert not np.isnan(feed_imported).any(), 'feed_imported must not be nan'
        assert not np.isnan(sup_feed_cost).any(), 'sup_feed_cost must not be nan'
        assert not np.isnan(new_state).any(), 'new_state must not be nan'
        return new_state, feed_imported, sup_feed_cost

    def supplemental_action_last(self, i_month, month, day, current_state, current_feed,
                                 current_money):
        """
        supplemental actions that might be needed for sub models at the end of each time step.  This is a placeholder to allow rapid development without modifying the base class. it is absolutely reasonable to leave as is.

        Here we use this function to save the cow buckets to output

        :param i_month:
        :param month:
        :param day:
        :param current_state:
        :param current_feed:
        :param current_money:
        :return:
        """
        # save the cow buckets to output
        self.out_lactating_cow_fraction[i_month] = self.lactating_cow_fraction
        self.out_dry_cow_fraction[i_month] = self.dry_cow_fraction
        self.out_replacement_fraction[i_month] = self.replacement_fraction

        # check replacement fraction
        if month in [5, 6, 7, 8, 9, 10, 11]:
            assert (self.replacement_fraction == self.replacement_rate).all(), (
                f'replacement fraction must be {self.replacement_rate=} in May, June, July, '
                f'August, September, October, November. Got {self.replacement_fraction}')
        else:
            assert (self.replacement_fraction == self.replacement_rate * 2).all(), (
                f'replacement fraction must be {2 * self.replacement_rate=} in Dec, Jan, Feb, March, April. '
                f'Got {self.replacement_fraction}')

        assert (self.lactating_cow_fraction + self.dry_cow_fraction <= self.max_cow[month]).all(), (
            f'total cows must be <= 1, got {self.lactating_cow_fraction + self.dry_cow_fraction}')

        assert (self.lactating_cow_fraction + self.dry_cow_fraction >= self.min_cow[month]).all(), (
            f'total cows must be >= {self.min_cow[month]}, '
            f'got {self.lactating_cow_fraction + self.dry_cow_fraction}')

        if month in [5, 6, 7]:
            assert (self.lactating_cow_fraction == 0).all(), (
                f'lactating cows must be 0 in May, June, July got {self.lactating_cow_fraction}')

        # cow loss per year applied pro-rata on the last day of the month.
        if day == month_len[month]:
            self.lactating_cow_fraction = self.lactating_cow_fraction * (1 - self.lactating_cow_loss / 12)
            self.lactating_cow_fraction[self.lactating_cow_fraction < 0] = 0
            self.dry_cow_fraction = self.dry_cow_fraction * (1 - self.dry_cow_loss / 12)
            self.dry_cow_fraction[self.dry_cow_fraction < 0] = 0
            self.replacement_fraction = self.replacement_fraction * (1 - self.replacement_cow_loss / 12)
            self.replacement_fraction[self.replacement_fraction < 0] = 0

        if month == 7 and day == month_len[7]:
            self.lactating_cow_fraction = self.dry_cow_fraction
            self.dry_cow_fraction = np.zeros(self.nsims)

        # add new replacements at end of November
        if month == 11 and day == month_len[11]:
            self.replacement_fraction = self.replacement_fraction + self.replacement_rate

        # dryoff and cull cows to replacement rate at end of April
        if month == 4 and day == month_len[4]:
            self.replacement_fraction = np.zeros(self.nsims) + self.replacement_rate
            self.dry_cow_fraction = np.zeros(self.nsims) + self.max_cow[5]
            self.lactating_cow_fraction = np.zeros(self.nsims)

        return current_state, current_feed, current_money

    def calc_marginal_cost_benefit(self, i_month, month, current_state, current_feed):
        """
        calculate the marginal cost and benefit of a potential state change

        :param i_month: model step
        :param month: integer month
        :param current_state: the current states of all models
        :return:
        """
        current_year = self.all_year[i_month]
        temp = self.all_year == current_year + 1
        if temp.any():
            stop_idx = np.argmax(temp)
        else:
            stop_idx = len(self.all_year)
        remaining_months = self.all_months[i_month:stop_idx]

        use_product_price, use_feed_cost, use_pg = self._calc_expected_pg_feed_prod_price(i_month, stop_idx,
                                                                                          remaining_months, month)

        # future for the current state
        cur_future_product, cur_sup_cost = self._calc_current_state_future(
            i_month=i_month,
            remaining_months=remaining_months,
            current_state=current_state,
            use_pg=use_pg,
            current_feed=current_feed,
            use_prod_price=use_product_price,
            use_feed_cost=use_feed_cost
        )

        # future for the alternate state
        (alt_fut_prod, alt_sup_cost, next_action, alt_lact_fraction,
         alt_dry_fraction) = self._calc_alternate_state_future(
            month=month,
            i_month=i_month,
            remaining_months=remaining_months,
            current_state=current_state,
            use_pg=use_pg,
            current_feed=current_feed,
            use_feed_cost=use_feed_cost,
            use_prod_price=use_product_price,
        )

        # calculate marginal cost
        marginal_cost = cur_future_product - alt_fut_prod
        assert (marginal_cost >= 0).all()
        self.out_marginal_cost[i_month] = marginal_cost

        # calculate marginal benefit
        marginal_benefit = cur_sup_cost - alt_sup_cost
        assert (marginal_benefit >= 0).all()
        self.out_marginal_benefit[i_month] = marginal_benefit

        return marginal_cost, marginal_benefit, next_action, alt_lact_fraction, alt_dry_fraction

    def _calc_expected_pg_feed_prod_price(self, i_month, stop_idx, remaining_months, month):
        """
        calculate the expected pasture growth, feed cost, and product price for the remaining months for the marginal cost/benefit calculation
        :param i_month:
        :param stop_idx:
        :param remaining_months:
        :param month:
        :return:
        """
        if self.marginal_cost_from_true:
            use_product_price = self.product_price[i_month:stop_idx]
        else:
            use_product_price = np.zeros(len(remaining_months), self.nsims)
            for i in range(month, 13):
                idx = remaining_months == i
                use_product_price[idx] = self.mean_prod_price[i]
        if self.marginal_benefit_from_true:
            use_pg = self.pg[i_month:stop_idx]

        else:
            use_pg = np.zeros(len(remaining_months), self.nsims)
            for i in range(month, 13):
                idx = remaining_months == i
                use_pg[idx] = self.mean_pg[i]

        use_pg = use_pg.copy()

        if self.marginal_benefit_from_true:
            use_feed_cost = self.sup_feed_cost[i_month:stop_idx]
        else:
            use_feed_cost = np.zeros(len(remaining_months), self.nsims)
            for i in range(month, 13):
                idx = remaining_months == i
                use_feed_cost[idx] = self.mean_sup_cost[i]

        use_feed_cost = use_feed_cost.copy()
        return use_product_price, use_feed_cost, use_pg

    def _calc_current_state_future(self, i_month, remaining_months, current_state, use_pg, current_feed,
                                   use_prod_price, use_feed_cost):

        current_1day_idxs = np.in1d(current_state, self.milk_1_aday_states)
        future_product, supplement_cost, deficit_feed = self._calc_cost_production(
            remaining_months=remaining_months,
            lact_fraction=self.lactating_cow_fraction,
            dry_fraction=self.dry_cow_fraction,
            once_aday_idx=current_1day_idxs,
            use_prod_price=use_prod_price,
            use_pg=use_pg,
            current_state=current_state,
            current_feed=current_feed,
            use_feed_cost=use_feed_cost,
            nsims=self.nsims)

        self.out_cur_fut_prod[i_month] = future_product
        self.out_cur_fut_feed_def[i_month] = deficit_feed

        return future_product, supplement_cost

    def _calc_alternate_state_future(self, month, i_month, remaining_months, current_state, use_pg, use_prod_price,
                                     current_feed, use_feed_cost):

        # can do one of once a day, cull, or dryoff

        next_action = self.calc_next_state_quant(month, current_state)
        to_once_aday_idx = next_action == 3
        once_aday_idx = np.in1d(current_state, self.milk_1_aday_states)
        no_change_idx = next_action == 0
        cull_idx = next_action == 1
        dryoff_idx = next_action == 2
        assert ((to_once_aday_idx + no_change_idx + cull_idx + dryoff_idx) == 1).all()

        # calculate best cow numbers
        alt_lact_fraction = np.zeros(self.nsims) - 1
        alt_dry_fraction = np.zeros(self.nsims) - 1

        # handle the 1 a day and no change states
        alt_lact_fraction[to_once_aday_idx] = self.lactating_cow_fraction[to_once_aday_idx]
        alt_dry_fraction[to_once_aday_idx] = self.dry_cow_fraction[to_once_aday_idx]
        alt_lact_fraction[no_change_idx] = self.lactating_cow_fraction[no_change_idx]
        alt_dry_fraction[no_change_idx] = self.dry_cow_fraction[no_change_idx]

        # cull cows optimisation
        new_cull_lact, new_cull_dry = [], []
        for i in np.where(cull_idx)[0]:
            tlact, tdry = self._cull_optimisation(
                lactating_cow=self.lactating_cow_fraction[[i]], dry_cow=self.dry_cow_fraction[[i]],
                month=month, remaining_months=remaining_months, once_aday_idx=once_aday_idx[[i]],
                use_prod_price=use_prod_price[:, [i]], use_pg=use_pg[:, [i]],
                current_state=current_state[[i]], current_feed=current_feed[[i]],
                use_feed_cost=use_feed_cost[:, [i]]
            )
            new_cull_lact.append(tlact)
            new_cull_dry.append(tdry)
        alt_lact_fraction[cull_idx] = new_cull_lact
        alt_dry_fraction[cull_idx] = new_cull_dry

        # dryoff cows optimisation
        new_dryoff_lact, new_dryoff_dry = [], []
        for i in np.where(dryoff_idx)[0]:
            tlact, tdry = self._dryoff_optmisation(
                lactating_cow=self.lactating_cow_fraction[[i]], dry_cow=self.dry_cow_fraction[[i]],
                remaining_months=remaining_months, once_aday_idx=once_aday_idx[[i]],
                use_prod_price=use_prod_price[:, [i]], use_pg=use_pg[:, [i]],
                current_state=current_state[[i]], current_feed=current_feed[[i]],
                use_feed_cost=use_feed_cost[:, [i]]
            )
            new_dryoff_lact.append(tlact)
            new_dryoff_dry.append(tdry)

        alt_lact_fraction[dryoff_idx] = new_dryoff_lact
        alt_dry_fraction[dryoff_idx] = new_dryoff_dry

        assert (alt_lact_fraction >= 0).all()
        assert (alt_dry_fraction >= 0).all()

        future_product, supplement_cost, deficit_feed = self._calc_cost_production(
            remaining_months=remaining_months,
            lact_fraction=alt_lact_fraction,
            dry_fraction=alt_dry_fraction,
            once_aday_idx=once_aday_idx | to_once_aday_idx,
            use_prod_price=use_prod_price,
            use_pg=use_pg,
            current_state=current_state,
            current_feed=current_feed,
            use_feed_cost=use_feed_cost,
            nsims=self.nsims)

        # feed requirements
        self.out_alt_fut_prod[i_month] = future_product
        self.out_cur_fut_feed_def[i_month] = deficit_feed
        assert (alt_lact_fraction >= 0).all()
        assert (alt_dry_fraction >= 0).all()

        return future_product, supplement_cost, next_action, alt_lact_fraction, alt_dry_fraction

    def calc_next_state_quant_1aday(self, month, current_state):
        """
        calculate the possible next state down (lower feed requirements) from  the current state

        :param current_state: the current state for all farms
        :return: actions np.ndarray shape (nsims,) of the actions:

        * 0: no change
        * 1: cull cows
        * 2: dryoff cows
        * 3: go to once a day milking
        """
        # can do one of once a day, cull, or dryoff not multiple

        if month in [5, 6, 7]:
            out = np.zeros(self.nsims, dtype=int)
        elif month in [8, 9, 10, 11, 12, 1, 2]:
            out = np.zeros(self.nsims, dtype=int) - 1
            # if feed runs low before the end of February go to
            # 1. once-a-day milking
            # 2. this cull up to the replacement rate (no on-going feed requirement)
            # 3. Post this dry-off (significantly reduced feed demand).
            once_idx = np.in1d(current_state, self.milk_1_aday_states)
            out[~once_idx] = 3
            cow_nums = self.lactating_cow_fraction + self.dry_cow_fraction
            cull_idx = (cow_nums > (self.min_cow[month] * 1.01)) & once_idx
            out[cull_idx] = 1
            dry_idx = (cow_nums <= (self.min_cow[month] * 1.01)) & (self.lactating_cow_fraction > 0) & once_idx
            out[dry_idx] = 2
            no_change_idx = (cow_nums <= (self.min_cow[month] * 1.01)) & (self.lactating_cow_fraction <= 0) & once_idx
            out[no_change_idx] = 0

        elif month in [3, 4]:
            # If feed runs low after end of February:
            # 1. Cull up to the replacement rate
            # 2. Post this go to once-a-day milking
            # 3. dryoff
            out = np.zeros(self.nsims, dtype=int) - 1
            cow_nums = self.lactating_cow_fraction + self.dry_cow_fraction
            cull_idx = (cow_nums > self.min_cow[month])
            out[cull_idx] = 1
            once_idx = np.in1d(current_state, self.milk_1_aday_states)
            out[~once_idx & ~cull_idx] = 3
            dry_idx = (cow_nums <= self.min_cow[month]) & (self.lactating_cow_fraction > 0) & once_idx
            out[dry_idx] = 2
            no_change_idx = (cow_nums <= self.min_cow[month]) & (self.lactating_cow_fraction <= 0) & once_idx
            out[no_change_idx] = 0
        else:
            raise ValueError('should not get here')
        assert out.min() >= 0, f'out must be >= 0, got {out.min()}'
        return out

    def calc_next_state_quant_not_1aday(self, month, current_state):
        if month in [5, 6, 7]:
            out = np.zeros(self.nsims, dtype=int)
        else:
            out = np.zeros(self.nsims, dtype=int) - 1
            # if feed runs low before the end of February go to
            # 1. once-a-day milking
            # 2. this cull up to the replacement rate (no on-going feed requirement)
            # 3. Post this dry-off (significantly reduced feed demand).
            cow_nums = self.lactating_cow_fraction + self.dry_cow_fraction
            cull_idx = (cow_nums > (self.min_cow[month] * 1.01))
            out[cull_idx] = 1
            dry_idx = (cow_nums <= (self.min_cow[month] * 1.01)) & (self.lactating_cow_fraction > 0)
            out[dry_idx] = 2
            no_change_idx = (cow_nums <= (self.min_cow[month] * 1.01)) & (self.lactating_cow_fraction <= 0)
            out[no_change_idx] = 0

        assert out.min() >= 0, f'out must be >= 0, got {out.min()}'
        return out

    def _cull_optimisation(self, lactating_cow, dry_cow, month, remaining_months, once_aday_idx,
                           use_prod_price, use_pg, current_state, current_feed, use_feed_cost):
        """
        how many cows to cull to get to the best option
        :return:
        """
        max_cull = lactating_cow + dry_cow - self.min_cow[month]
        assert (max_cull > 0).all()

        # check dimensions
        assert lactating_cow.shape == (1,)
        assert dry_cow.shape == (1,)
        assert once_aday_idx.shape == (1,)
        assert use_prod_price.shape == (len(remaining_months), 1)
        assert use_pg.shape == (len(remaining_months), 1)
        assert current_state.shape == (1,)
        assert current_feed.shape == (1,)
        assert use_feed_cost.shape == (len(remaining_months), 1)

        def cull_prod(ncull):
            rm_dry = np.min(np.concatenate((dry_cow, ncull)))
            new_dry_fract = dry_cow - rm_dry
            new_lact_fract = lactating_cow - (ncull - rm_dry)
            future_product, supplement_cost, deficit_feed = self._calc_cost_production(
                remaining_months=remaining_months,
                lact_fraction=new_lact_fract,
                dry_fraction=new_dry_fract,
                once_aday_idx=once_aday_idx,
                use_prod_price=use_prod_price,
                use_pg=use_pg,
                current_state=current_state,
                current_feed=current_feed,
                use_feed_cost=use_feed_cost,
                nsims=1)
            return (future_product[0] - supplement_cost[0]) * -1

        res = minimize_scalar(cull_prod, bounds=(0, max_cull))

        cull = res.x
        assert res.success, f'optimisation failed {res.message}'
        rm_dry = np.min(np.concatenate((dry_cow, [cull])))
        new_dry_fract = dry_cow - rm_dry
        new_lact_fract = lactating_cow - (cull - rm_dry)
        return new_lact_fract[0], new_dry_fract[0]

    def _dryoff_optmisation(self, lactating_cow, dry_cow, remaining_months, once_aday_idx,
                            use_prod_price, use_pg, current_state, current_feed, use_feed_cost):
        """
        how many cows to cull to get to the best option
        :return:
        """
        max_dry = lactating_cow
        assert (max_dry > 0).all()

        # check dimensions
        assert lactating_cow.shape == (1,)
        assert dry_cow.shape == (1,)
        assert once_aday_idx.shape == (1,)
        assert use_prod_price.shape == (len(remaining_months), 1)
        assert use_pg.shape == (len(remaining_months), 1)
        assert current_state.shape == (1,)
        assert current_feed.shape == (1,)
        assert use_feed_cost.shape == (len(remaining_months), 1)

        def cull_prod(ndry, rt_both=False):
            new_dry_fract = dry_cow + ndry
            new_lact_fract = lactating_cow - ndry
            future_product, supplement_cost, deficit_feed = self._calc_cost_production(
                remaining_months=remaining_months,
                lact_fraction=new_lact_fract,
                dry_fraction=new_dry_fract,
                once_aday_idx=once_aday_idx,
                use_prod_price=use_prod_price,
                use_pg=use_pg,
                current_state=current_state,
                current_feed=current_feed,
                use_feed_cost=use_feed_cost,
                nsims=1)
            if rt_both:
                return future_product[0], supplement_cost[0]
            return (future_product[0] - supplement_cost[0]) * -1

        res = minimize_scalar(cull_prod, bounds=(0, max_dry))
        dry = res.x
        new_dry_fract = dry_cow + dry
        new_lact_fract = lactating_cow - dry
        return new_lact_fract[0], new_dry_fract[0]

    def _calc_cost_production(self, remaining_months, lact_fraction, dry_fraction, once_aday_idx, use_prod_price,
                              use_pg, current_state, current_feed, use_feed_cost, nsims):
        remaining_months = remaining_months.copy()
        lact_fraction = lact_fraction.copy()
        dry_fraction = dry_fraction.copy()
        once_aday_idx = once_aday_idx.copy()
        use_prod_price = use_prod_price.copy()
        use_pg = use_pg.copy()
        current_state = current_state.copy()
        current_feed = current_feed.copy()
        use_feed_cost = use_feed_cost.copy()

        assert len(remaining_months) > 0, 'remaining_months must be > 0'
        future_lactating_cow_fraction = np.zeros((len(remaining_months), nsims))
        future_dry_cow_fraction = np.zeros((len(remaining_months), nsims))
        future_replacement_fraction = np.zeros((len(remaining_months), nsims))
        for i, m in enumerate(set(remaining_months)):
            idx = remaining_months == m
            lc_with_loss = lact_fraction * (1 - self.lactating_cow_loss / 12 * i)
            dc_with_loss = dry_fraction * (1 - self.dry_cow_loss / 12 * i)
            if m in [5, 6, 7]:
                future_lactating_cow_fraction[idx] = 0
                future_dry_cow_fraction[idx] = lc_with_loss + dc_with_loss
            else:
                future_lactating_cow_fraction[idx] = lc_with_loss
                future_dry_cow_fraction[idx] = dc_with_loss

            if m in [12, 1, 2, 3, 4]:
                future_replacement_fraction[idx] = 2 * self.replacement_rate
            else:
                future_replacement_fraction[idx] = self.replacement_rate

        future_product = (
                np.array([self.kgms_per_lac_cow[m] for m in remaining_months])[:, np.newaxis]
                * (self.peak_lact_cow_per_ha * future_lactating_cow_fraction).round(self.precision)
        )
        future_product[:, once_aday_idx] *= self.one_a_daymilk_production_fraction

        future_product = future_product * use_prod_price
        future_product = future_product.sum(axis=0)

        # feed requirements
        feed_needed_lact = np.zeros((len(remaining_months), nsims))
        feed_needed_dry = np.zeros((len(remaining_months), nsims))
        feed_needed_replacement = np.zeros((len(remaining_months), nsims))

        # need to move to for loop otherwise it won't use the correct values
        for i, m in enumerate(set(remaining_months)):
            idx = remaining_months == m
            feed_needed_lact[idx] = self.feed_per_cow_lactating[m] * future_lactating_cow_fraction[
                idx] * self.peak_lact_cow_per_ha
            feed_needed_dry[idx] = self.feed_per_cow_dry[m] * future_dry_cow_fraction[idx] * self.peak_lact_cow_per_ha
            feed_needed_replacement[idx] = self.feed_per_cow_replacement[m] * future_replacement_fraction[
                idx] * self.peak_lact_cow_per_ha

        feed_needed_lact[:, once_aday_idx] *= self.feed_fraction_1aday
        total_feed_needed = feed_needed_lact + feed_needed_dry + feed_needed_replacement

        current_home_production = self.convert_pg_to_me(use_pg, current_state) * self.homegrown_store_efficiency
        current_stored_feed = current_feed * self.supplemental_efficiency
        deficit_feed = np.zeros(total_feed_needed.shape)
        for i in range(len(deficit_feed)):
            home_prod = current_home_production[i]
            current_stored_feed += home_prod
            feed_need = total_feed_needed[i]
            temp = np.concatenate((np.zeros(current_stored_feed.shape)[:, np.newaxis],
                                   (feed_need - current_stored_feed)[:, np.newaxis]), axis=1)
            deficit = np.max(temp, axis=1)
            deficit_feed[i] = deficit
            current_stored_feed -= feed_need - deficit

        assert (deficit_feed >= 0).all()

        supplement_needed = deficit_feed / self.supplemental_efficiency
        supplement_cost = (supplement_needed * use_feed_cost + self.sup_feedout_cost * supplement_needed).sum(axis=0)
        return future_product, supplement_cost, deficit_feed.sum(axis=0)
