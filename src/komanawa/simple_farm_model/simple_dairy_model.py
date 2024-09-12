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

The marginal cost and benefit of a potential state change is assessed on the 1st and 15th day of each month.

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

The model assesses the alternate step change in the following order:

1.  Before end of February (months 8-2) go to:
    1. this cull up to the replacement rate (no on-going feed requirement)
    2. Post this dry-off (significantly reduced feed demand).
2. After end of Feb, but before end of April (months 3-4) go to:
    1. Cull up to the replacement rate
    2. dry-off stock
3. For months 5, 6, 7, all cows are dried off regardless so
    1. No change

The amount of cows to cull/dry off can either be optimised (cull_dry_step=None) or set to a fixed step size (cull_dry_step=0.01).  The optimisation is done using the scipy minimize_scalar function. Note that the optimisation process significantly increases the runtime of the model.
"""
import logging
import numpy as np
import pandas as pd
from numpy.core.numeric import allclose
from scipy.optimize import minimize_scalar
from copy import deepcopy
import matplotlib.pyplot as plt
from komanawa.simple_farm_model.run_multiprocess import run_multiprocess
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel, month_len
from komanawa.simple_farm_model.utils import leclose, geclose

mj_per_kg_dm = 11  # MJ ME /kg DM
monly_ms_prod = {  # kgMS per lactating cow per month
    1: 57.47,
    2: 49.10,
    3: 50.13,
    4: 45.89,
    5: 0,  # based on discussions with andrew curtis, not worth including may prod. prev. 45.42, # todo check this
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
lactating_feed = {k: v * 123.19 for k, v in daily_ms_prod.items()}  # assume 123.19 MJ ME/kg MS

default_peak_cow = (3.48 + 2.82) / 2 / 1.44  # (dairy platform density + replacement density) / 2 / 1.44


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
    annual_supplement_import = 0.0  # 0% of total feed required per year (less any stored feed)

    ndays_feed_import = 7  # when needing to import feed, import 7 days worth of feed (from 7 days supplemental needed)
    state_change_days = (1, 15)  # allow for a change in farm state on day 1 and 15 of each month
    month_reset = 7  # trigger farm reset on day 1 in August,
    homegrown_efficiency = 0.8
    supplemental_efficiency = 0.9
    sup_feedout_cost = 120 / 1000 / mj_per_kg_dm  # 120/ton DM --> convert to ME, MJ
    homegrown_store_efficiency = 1  # todo possibly set this lower...
    homegrown_storage_cost = 175 / 1000 / mj_per_kg_dm  # 175/ton/DM --> convert to ME, MJ
    one_a_daymilk_production_fraction = 1 - 0.13  # 13% less milk production on 1-a-day
    _start_lactating_cow_fraction = 0  # % of cows are lactating on day 1  (july)#
    _start_dry_cow_fraction = 1  # 100% of cows are dry on day 1 (july)
    _start_replacement_fraction = None  # set internally to replacement rate
    _annual_feed_needed = None  # set at first reset
    replacement_rate = 0.22  # 22% of cows are replacements
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
    alt_kwargs = (
    'peak_lact_cow_per_ha', 'cull_dry_step', 'ncore_opt', 'logging_level', 'opt_mode', 'cull_levels', 'dryoff_levels')

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price,
                 opt_mode='optimised', monthly_input=True,
                 peak_lact_cow_per_ha=default_peak_cow, ncore_opt=1, logging_level=logging.CRITICAL,
                 cull_dry_step=None, cull_levels=None, dryoff_levels=None):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number float or np.ndarray shape (nsims,) MJ/ha
        :param imoney: initial money number float or np.ndarray shape (nsims,) $/ha
        :param sup_feed_cost: cost of supplementary feed $/MJ float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param product_price: income price $/kg product float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param opt_mode: one of:

        * 'optimised': use scipy.minimize_scalar to optimise the fraction of cows to cull dryoff at each decision point (cpu intensive, but most precise, low memory) cull_dry_step must be None
        * 'step': use a fixed step size for cull/dryoff decision (less cpu intensive, less precise, low memory) cull_dry_step must be set
        * 'coarse': pick the best of an a. priori. number of cull/dryoff decisions (low cpu, high memory, moderate precision) cull_dry_step must be None

        :param monthly_input: if True, monthly input, if False, daily input (365 days per year)
        :param peak_lact_cow_per_ha: peak lactating cows per ha float
        :param ncore_opt: number of cores to use for optimization of the cull/dryoff decision if ncore_opt==1, do not multiprocess, if ncore_opt > 1, use ncore_opt cores, if ncore_opt =None, use all available cores
        :param logging_level: logging level for the multiprocess component
        :param cull_dry_step: if None, use the optimized cull/dryoff decision, if float, use the cull/dryoff decision with a step size of cull_dry_step
        :param cull_levels: if opt_mode='coarse' the number of equally spaced cull steps to make, otherwise must be None
        :param dryoff_levels: if opt_mode='coarse' the number of equally spaced dryoff steps to make, otherwise must be None
        """
        assert opt_mode in ('optimised', 'step', 'coarse'), 'opt_mode must be one of: optimised, step, coarse'
        if opt_mode == 'optimised':
            assert cull_dry_step is None
            assert cull_levels is None
            assert dryoff_levels is None
            self.cull_levels = None
            self.dryoff_levels = None
            self.cull_dry_step = None
            self._calc_alternate_state_future = self._calc_alternate_state_future_opt
        elif opt_mode == 'step':
            assert cull_levels is None
            assert dryoff_levels is None
            self.cull_levels = None
            self.dryoff_levels = None
            assert pd.api.types.is_float(cull_dry_step), 'cull_dry_step must be a float'
            assert 0 <= cull_dry_step <= 1, 'cull_dry_step must be between 0 and 1'
            self.cull_dry_step = float(cull_dry_step)
            self._calc_alternate_state_future = self._calc_alternate_state_future_step
        elif opt_mode == 'coarse':
            assert pd.api.types.is_integer(cull_levels), 'cull_levels must be an integer'
            assert pd.api.types.is_integer(dryoff_levels), 'dryoff_levels must be an integer'
            self.cull_levels = int(cull_levels)
            self.dryoff_levels = int(dryoff_levels)
            assert cull_dry_step is None
            self.cull_dry_step = None
            self._calc_alternate_state_future = self._calc_alternate_state_coarse_opt
        else:
            raise ValueError(f'opt_mode must be one of: optimised, step, coarse, not {opt_mode}')

        self.opt_mode = opt_mode
        self.ncore_opt = ncore_opt
        self.logging_level = logging_level
        assert peak_lact_cow_per_ha > 0, 'peak_lact_cow_per_ha must be >0'
        self.peak_lact_cow_per_ha = float(peak_lact_cow_per_ha)
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
        self.get_annual_feed()
        self.reset_state(0, self.model_feed[0], self.model_money[0])

        # identify which states are 1 a day milking
        self.milk_1_aday_states = [i for i, a_day_2 in self.states.items() if not a_day_2]

        self.set_mean_pg(pg_dict=None, pg_raw=self.pg, months=self.all_months)
        self.set_mean_sup_cost(sup_dict=None, sup_cost_raw=self.sup_feed_cost, months=self.all_months)
        self.set_mean_product_price(prod_price_dict=None, prod_price_raw=self.product_price, months=self.all_months)

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
        self.obj_dict['feed_scarcity_cost'] = 'feed_scarcity_cost'

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
        self.long_name_dict['feed_scarcity_cost'] = 'Feed Scarcity Cost ($)'

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
        self.feed_scarcity_cost = np.zeros(self.model_shape) * np.nan

    def calc_feed_scarcity_cost(self, i_month, start_cum_ann_feed_import, new_feed, use_sup_feed_cost,
                                nsims):
        """
        Calculate the feed scarcity cost based on the previous cumulative feed import and the new feed import

        :param start_cum_ann_feed_import: previous cumulative feed import (one per sim)
        :param new_feed: new feed import (ndays, nsims)
        :return: feed scarcity cost ($ for the time period) (NOT $/MJ !) np.ndarray shape (ndays, nsims)
        """
        assert start_cum_ann_feed_import.shape == new_feed.shape[1:] == (
            nsims,), f'{start_cum_ann_feed_import.shape=} {new_feed.shape=}'

        return np.zeros(new_feed.shape)

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
        annual_feed = self.get_annual_feed()

        import_feed = annual_feed * self.annual_supplement_import - current_feed
        import_feed[import_feed < 0] = 0
        supplement_cost = import_feed * self.sup_feed_cost[i_month]
        scarcity_cost = self.calc_feed_scarcity_cost(i_month, np.zeros(self.nsims),
                                                     import_feed[np.newaxis],
                                                     use_sup_feed_cost=self.sup_feed_cost,
                                                     nsims=self.nsims)[0]
        self.feed_scarcity_cost[i_month, :] = np.nansum(np.concatenate((
            scarcity_cost[np.newaxis],
            self.feed_scarcity_cost[[i_month], :]), axis=0), axis=0)
        self.model_feed_imported[i_month, :] = np.nansum(np.concatenate((import_feed[np.newaxis],
                                                                         self.model_feed_imported[[i_month], :]),
                                                                        axis=0), axis=0)
        self.model_feed_cost[i_month, :] = np.nansum(np.concatenate((supplement_cost[np.newaxis],
                                                                     scarcity_cost[np.newaxis],
                                                                     self.model_feed_cost[[i_month], :]),
                                                                    axis=0), axis=0)
        current_feed += import_feed
        current_money -= supplement_cost + scarcity_cost

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
        scarcity_cost = self.calc_feed_scarcity_cost(
            i_month,
            start_cum_ann_feed_import=self.cum_feed_import[i_month - 1],
            new_feed=feed_imported[np.newaxis],
            use_sup_feed_cost=self.sup_feed_cost[i_month: i_month + feed_imported.shape[0]],
            nsims=self.nsims)[0]
        sup_feed_cost = feed_imported * self.sup_feed_cost[i_month] + scarcity_cost
        self.feed_scarcity_cost[i_month, :] = np.nansum(np.concatenate((
            scarcity_cost[np.newaxis],
            self.feed_scarcity_cost[[i_month], :]), axis=0), axis=0)
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
            assert np.allclose(self.replacement_fraction, self.replacement_rate), (
                f'replacement fraction must be {self.replacement_rate=} in May, June, July, '
                f'August, September, October, November. Got {self.replacement_fraction}')
        else:
            assert np.allclose(self.replacement_fraction, self.replacement_rate * 2), (
                f'replacement fraction must be {2 * self.replacement_rate=} in Dec, Jan, Feb, March, April. '
                f'Got {self.replacement_fraction}')

        assert leclose(self.lactating_cow_fraction + self.dry_cow_fraction, self.max_cow[month]).all(), (
            f'total cows must be <= 1, got {self.lactating_cow_fraction + self.dry_cow_fraction}')

        assert geclose(self.lactating_cow_fraction + self.dry_cow_fraction, self.min_cow[month]).all(), (
            f'total cows must be >= {self.min_cow[month]}, '
            f'got {self.lactating_cow_fraction + self.dry_cow_fraction}')

        if month in [5, 6, 7]:
            assert allclose(self.lactating_cow_fraction, 0), (
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
            use_feed_cost=use_feed_cost,
            current_cum_import=self.cum_feed_import[i_month - 1],
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
            cum_feed=self.cum_feed_import[i_month - 1]
        )

        # calculate marginal cost
        marginal_cost = cur_future_product - alt_fut_prod
        assert geclose(marginal_cost, 0).all()
        self.out_marginal_cost[i_month] = marginal_cost

        # calculate marginal benefit
        marginal_benefit = cur_sup_cost - alt_sup_cost
        assert geclose(marginal_benefit, 0).all()
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
                                   use_prod_price, use_feed_cost, current_cum_import):

        current_1day_idxs = np.in1d(current_state, self.milk_1_aday_states)
        future_product, supplement_cost, deficit_feed = self._calc_cost_production(
            i_month=i_month,
            remaining_months=remaining_months,
            lact_fraction=self.lactating_cow_fraction,
            dry_fraction=self.dry_cow_fraction,
            once_aday_idx=current_1day_idxs,
            use_prod_price=use_prod_price,
            use_pg=use_pg,
            current_state=current_state,
            current_feed=current_feed,
            use_feed_cost=use_feed_cost,
            current_cum_import=current_cum_import,
            nsims=self.nsims)

        self.out_cur_fut_prod[i_month] = future_product
        self.out_cur_fut_feed_def[i_month] = deficit_feed

        return future_product, supplement_cost

    def _prep_alt_data(self, month, current_state):
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
        return (cull_idx, dryoff_idx, once_aday_idx, alt_lact_fraction, alt_dry_fraction,
                to_once_aday_idx, next_action)

    def _run_alt(self, alt_lact_fraction, alt_dry_fraction, i_month, remaining_months, once_aday_idx, to_once_aday_idx,
                 use_pg,
                 use_prod_price, current_state, current_feed, use_feed_cost, cum_feed):
        assert geclose(alt_lact_fraction, 0).all()
        assert geclose(alt_dry_fraction, 0).all()

        future_product, supplement_cost, deficit_feed = self._calc_cost_production(
            i_month=i_month,
            remaining_months=remaining_months,
            lact_fraction=alt_lact_fraction,
            dry_fraction=alt_dry_fraction,
            once_aday_idx=once_aday_idx | to_once_aday_idx,
            use_prod_price=use_prod_price,
            use_pg=use_pg,
            current_state=current_state,
            current_feed=current_feed,
            use_feed_cost=use_feed_cost,
            current_cum_import=cum_feed,
            nsims=self.nsims)

        # feed requirements
        self.out_alt_fut_prod[i_month] = future_product
        self.out_cur_fut_feed_def[i_month] = deficit_feed
        assert geclose(alt_lact_fraction, 0).all()
        assert geclose(alt_dry_fraction, 0).all()

        return future_product, supplement_cost, alt_lact_fraction, alt_dry_fraction

    def _calc_alternate_state_coarse_opt(self, month, i_month, remaining_months, current_state, use_pg, use_prod_price,
                                         current_feed, use_feed_cost, cum_feed):
        (cull_idx, dryoff_idx, once_aday_idx, alt_lact_fraction, alt_dry_fraction,
         to_once_aday_idx, next_action) = self._prep_alt_data(month, current_state)

        out_alt_fut_prod = np.zeros_like(alt_lact_fraction)
        out_cur_fut_feed_def = np.zeros_like(alt_lact_fraction)
        out_alt_sup_cost = np.zeros_like(alt_lact_fraction)

        if cull_idx.any():
            # shape of the new data (alt_cull) is (nsims_with_cull, ncull_steps)
            max_cull = (self.lactating_cow_fraction + self.dry_cow_fraction - self.min_cow[month])[cull_idx]
            ncull = np.array([np.linspace(0, m, self.cull_levels) for m in max_cull])
            ncull_org = ncull.copy()  # for debugging
            alt_cull_dry = np.repeat(self.dry_cow_fraction[cull_idx][:, np.newaxis], self.cull_levels, axis=1)
            alt_cull_lact = np.repeat(self.lactating_cow_fraction[cull_idx][:, np.newaxis], self.cull_levels, axis=1)
            t = np.maximum(0, alt_cull_dry - ncull)
            ncull = ncull - (alt_cull_dry - t)
            alt_cull_dry = t
            t0 = np.maximum(0, alt_cull_lact - ncull)
            ncull = ncull - (alt_cull_lact - t0)
            alt_cull_lact = t0
            assert np.allclose(ncull, 0)
            assert geclose((alt_cull_lact + alt_cull_dry), self.min_cow[month]).all()
            temp_future_product, temp_supplement_cost, temp_deficit_feed = self.__run_alt_coarse(
                in_alt_lact=alt_cull_lact,
                in_alt_dry=alt_cull_dry,
                i_month=i_month,
                remaining_months=remaining_months,
                once_aday_idx=once_aday_idx[cull_idx],
                to_once_aday_idx=to_once_aday_idx[cull_idx],
                in_pg=use_pg[:, cull_idx],
                in_prod_price=use_prod_price[:, cull_idx],
                current_state=current_state[cull_idx],
                current_feed=current_feed[cull_idx],
                in_feed_cost=use_feed_cost[:, cull_idx],
                cum_feed=cum_feed[cull_idx],
                steps=self.cull_levels
            )

            net = temp_future_product - temp_supplement_cost
            best_idx = np.argmax(net, axis=1)
            d1_idx = np.arange(alt_cull_lact.shape[0])
            alt_lact_fraction[cull_idx] = alt_cull_lact[d1_idx, best_idx]
            alt_dry_fraction[cull_idx] = alt_cull_dry[d1_idx, best_idx]
            out_alt_fut_prod[cull_idx] = temp_future_product[d1_idx, best_idx]
            out_cur_fut_feed_def[cull_idx] = temp_deficit_feed[d1_idx, best_idx]
            out_alt_sup_cost[cull_idx] = temp_supplement_cost[d1_idx, best_idx]

        if dryoff_idx.any():
            ndry = np.array([np.linspace(0, m, self.dryoff_levels) for m in self.lactating_cow_fraction[dryoff_idx]])
            alt_dry_dry = np.repeat(self.dry_cow_fraction[dryoff_idx][:, np.newaxis], self.dryoff_levels, axis=1)
            alt_dry_lact = np.repeat(self.lactating_cow_fraction[dryoff_idx][:, np.newaxis], self.dryoff_levels, axis=1)
            alt_dry_dry += ndry
            alt_dry_lact -= ndry
            temp_future_product, temp_supplement_cost, temp_deficit_feed = self.__run_alt_coarse(
                in_alt_lact=alt_dry_lact,
                in_alt_dry=alt_dry_dry,
                i_month=i_month,
                remaining_months=remaining_months,
                once_aday_idx=once_aday_idx[dryoff_idx],
                to_once_aday_idx=to_once_aday_idx[dryoff_idx],
                in_pg=use_pg[:, dryoff_idx],
                in_prod_price=use_prod_price[:, dryoff_idx],
                current_state=current_state[dryoff_idx],
                current_feed=current_feed[dryoff_idx],
                in_feed_cost=use_feed_cost[:, dryoff_idx],
                cum_feed=cum_feed[dryoff_idx],
                steps=self.dryoff_levels
            )

            net = temp_future_product - temp_supplement_cost
            best_idx = np.argmax(net, axis=1)
            d1_idx = np.arange(alt_dry_lact.shape[0])
            alt_lact_fraction[dryoff_idx] = alt_dry_lact[d1_idx, best_idx]
            alt_dry_fraction[dryoff_idx] = alt_dry_dry[d1_idx, best_idx]
            out_alt_fut_prod[dryoff_idx] = temp_future_product[d1_idx, best_idx]
            out_cur_fut_feed_def[dryoff_idx] = temp_deficit_feed[d1_idx, best_idx]
            out_alt_sup_cost[dryoff_idx] = temp_supplement_cost[d1_idx, best_idx]

        self.out_alt_fut_prod[i_month] = out_alt_fut_prod
        self.out_cur_fut_feed_def[i_month] = out_cur_fut_feed_def
        assert geclose(alt_lact_fraction, 0).all()
        assert geclose(alt_dry_fraction, 0).all()

        return out_alt_fut_prod, out_alt_sup_cost, next_action, alt_lact_fraction, alt_dry_fraction

    def __run_alt_coarse(self, in_alt_lact, in_alt_dry, i_month, remaining_months, once_aday_idx, to_once_aday_idx,
                         in_pg, in_prod_price, current_state, current_feed, in_feed_cost, cum_feed, steps):
        assert in_alt_lact.shape == in_alt_dry.shape
        datashape = in_alt_lact.shape
        in_alt_dry = in_alt_dry.flatten()
        in_alt_lact = in_alt_lact.flatten()

        # define and check input shapes
        in_2d_shape = (len(remaining_months), datashape[0])
        in_1d_shape = (datashape[0],)
        assert in_pg.shape == in_prod_price.shape == in_feed_cost.shape == in_2d_shape
        assert current_state.shape == current_feed.shape == cum_feed.shape == in_1d_shape
        assert once_aday_idx.shape == to_once_aday_idx.shape == in_1d_shape

        # reshape 1d input data
        use_once_day_idx = np.repeat(once_aday_idx[:, np.newaxis], steps, axis=1)
        assert use_once_day_idx.shape == datashape
        use_once_day_idx = use_once_day_idx.flatten()
        use_to_once_day_idx = np.repeat(to_once_aday_idx[:, np.newaxis], steps, axis=1)
        assert use_to_once_day_idx.shape == datashape
        use_to_once_day_idx = use_to_once_day_idx.flatten()
        use_current_state = np.repeat(current_state[:, np.newaxis], steps, axis=1)
        assert use_current_state.shape == datashape
        use_current_state = use_current_state.flatten()
        use_current_feed = np.repeat(current_feed[:, np.newaxis], steps, axis=1)
        assert use_current_feed.shape == datashape
        use_current_feed = use_current_feed.flatten()
        use_cum_feed = np.repeat(cum_feed[:, np.newaxis], steps, axis=1)
        assert use_cum_feed.shape == datashape
        use_cum_feed = use_cum_feed.flatten()

        # reshape 2d input data
        use_pg = np.repeat(in_pg[:, :, np.newaxis], steps, axis=2)
        assert use_pg.shape == remaining_months.shape + datashape
        use_pg = use_pg.reshape((use_pg.shape[0], use_pg.shape[1] * use_pg.shape[2]))
        use_prod_price = np.repeat(in_prod_price[:, :, np.newaxis], steps, axis=2)
        assert use_prod_price.shape == remaining_months.shape + datashape
        use_prod_price = use_prod_price.reshape(
            (use_prod_price.shape[0], use_prod_price.shape[1] * use_prod_price.shape[2]))
        use_feed_cost = np.repeat(in_feed_cost[:, :, np.newaxis], steps, axis=2)
        assert use_feed_cost.shape == remaining_months.shape + datashape
        use_feed_cost = use_feed_cost.reshape((use_feed_cost.shape[0], use_feed_cost.shape[1] * use_feed_cost.shape[2]))

        future_product, supplement_cost, deficit_feed = self._calc_cost_production(
            i_month=i_month,
            remaining_months=remaining_months,
            lact_fraction=in_alt_lact,
            dry_fraction=in_alt_dry,
            once_aday_idx=use_once_day_idx | use_to_once_day_idx,
            use_prod_price=use_prod_price,
            use_pg=use_pg,
            current_state=use_current_state,
            current_feed=use_current_feed,
            use_feed_cost=use_feed_cost,
            current_cum_import=use_cum_feed,
            nsims=np.prod(datashape))

        future_product = future_product.reshape(datashape)
        supplement_cost = supplement_cost.reshape(datashape)
        deficit_feed = deficit_feed.reshape(datashape)
        return future_product, supplement_cost, deficit_feed

    def _calc_alternate_state_future_step(self, month, i_month, remaining_months, current_state, use_pg, use_prod_price,
                                          current_feed, use_feed_cost, cum_feed):
        (cull_idx, dryoff_idx, once_aday_idx, alt_lact_fraction, alt_dry_fraction,
         to_once_aday_idx, next_action) = self._prep_alt_data(month, current_state)

        if cull_idx.any():
            ncull = np.full(cull_idx.shape, 0.)
            ncull[cull_idx] = self.cull_dry_step
            max_cull = self.lactating_cow_fraction + self.dry_cow_fraction - self.min_cow[month]
            ncull[ncull > max_cull] = max_cull[ncull > max_cull]
            alt_dry_fraction[cull_idx] = t = np.maximum(0, self.dry_cow_fraction[cull_idx] - ncull[cull_idx])
            ncull[cull_idx] = ncull[cull_idx] - (self.dry_cow_fraction[cull_idx] - t)
            alt_lact_fraction[cull_idx] = t = np.maximum(0, self.lactating_cow_fraction[cull_idx] - ncull[cull_idx])
            ncull[cull_idx] = ncull[cull_idx] - (self.lactating_cow_fraction[cull_idx] - t)
            assert np.allclose(ncull, 0)
            assert geclose(alt_lact_fraction[cull_idx] + alt_dry_fraction[cull_idx], self.min_cow[month]).all()

        if dryoff_idx.any():
            ndry = np.full(dryoff_idx.shape, 0)
            ndry[dryoff_idx] = self.cull_dry_step
            max_dry = self.lactating_cow_fraction
            ndry[ndry > max_dry] = max_dry[ndry > max_dry]
            alt_dry_fraction[dryoff_idx] = self.dry_cow_fraction[dryoff_idx] + ndry[dryoff_idx]
            alt_lact_fraction[dryoff_idx] = self.lactating_cow_fraction[dryoff_idx] - ndry[dryoff_idx]

        (future_product, supplement_cost, out_alt_lact,
         out_alt_dry) = self._run_alt(alt_lact_fraction, alt_dry_fraction, i_month, remaining_months, once_aday_idx,
                                      to_once_aday_idx, use_pg,
                                      use_prod_price, current_state, current_feed, use_feed_cost, cum_feed)
        return future_product, supplement_cost, next_action, out_alt_lact, out_alt_dry

    def _calc_alternate_state_future_opt(self, month, i_month, remaining_months, current_state, use_pg, use_prod_price,
                                         current_feed, use_feed_cost, cum_feed):
        # can do one of once a day, cull, or dryoff
        (cull_idx, dryoff_idx, once_aday_idx, alt_lact_fraction, alt_dry_fraction,
         to_once_aday_idx, next_action) = self._prep_alt_data(month, current_state)

        # cull cows optimisation
        if cull_idx.any():
            cull_runs = []
            for i in np.where(cull_idx)[0]:
                cull_runs.append(dict(iloc=i,
                                      i_month=i_month,
                                      lactating_cow=self.lactating_cow_fraction[[i]],
                                      dry_cow=self.dry_cow_fraction[[i]],
                                      month=month, remaining_months=remaining_months,
                                      once_aday_idx=once_aday_idx[[i]],
                                      use_prod_price=use_prod_price[:, [i]], use_pg=use_pg[:, [i]],
                                      current_state=current_state[[i]],
                                      current_feed=current_feed[[i]],
                                      use_feed_cost=use_feed_cost[:, [i]],
                                      cum_feed=cum_feed[[i]]
                                      ))
            if self.ncore_opt == 1:
                opt_data_cull = [self._cull_optimisation(**run) for run in cull_runs]
            else:
                opt_data_cull = run_multiprocess(func=self._cull_opt_mp, runs=cull_runs,
                                                 logging_level=self.logging_level,
                                                 num_cores=self.ncore_opt)

            # opt_data order iloc, new_lact_fract, new_dry_fract
            opt_data_cull = np.array(opt_data_cull)
            alt_lact_fraction[opt_data_cull[:, 0].astype(int)] = opt_data_cull[:, 1]
            alt_dry_fraction[opt_data_cull[:, 0].astype(int)] = opt_data_cull[:, 2]

        # dryoff cows optimisation
        if dryoff_idx.any():
            dryoff_runs = []
            for i in np.where(dryoff_idx)[0]:
                dryoff_runs.append(dict(iloc=i,
                                        i_month=i_month,
                                        lactating_cow=self.lactating_cow_fraction[[i]],
                                        dry_cow=self.dry_cow_fraction[[i]],
                                        remaining_months=remaining_months,
                                        once_aday_idx=once_aday_idx[[i]],
                                        use_prod_price=use_prod_price[:, [i]], use_pg=use_pg[:, [i]],
                                        current_state=current_state[[i]],
                                        current_feed=current_feed[[i]],
                                        use_feed_cost=use_feed_cost[:, [i]],
                                        cum_feed=cum_feed[[i]]
                                        ))

            if self.ncore_opt == 1:
                opt_data_dry = [self._dryoff_optmisation(**run) for run in dryoff_runs]
            else:
                opt_data_dry = run_multiprocess(func=self._dryoff_opt_mp, runs=dryoff_runs,
                                                logging_level=self.logging_level,
                                                num_cores=self.ncore_opt)
            opt_data_dry = np.array(opt_data_dry)
            alt_lact_fraction[opt_data_dry[:, 0].astype(int)] = opt_data_dry[:, 1]
            alt_dry_fraction[opt_data_dry[:, 0].astype(int)] = opt_data_dry[:, 2]

        (future_product, supplement_cost, out_alt_lact,
         out_alt_dry) = self._run_alt(alt_lact_fraction, alt_dry_fraction, i_month, remaining_months, once_aday_idx,
                                      to_once_aday_idx, use_pg,
                                      use_prod_price, current_state, current_feed, use_feed_cost, cum_feed)
        return future_product, supplement_cost, next_action, out_alt_lact, out_alt_dry

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
        assert geclose(out.min(), 0), f'out must be >= 0, got {out.min()}'
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

        assert geclose(out.min(), 0), f'out must be >= 0, got {out.min()}'
        return out

    def _cull_opt_mp(self, kwargs):
        return self._cull_optimisation(**kwargs)

    def _cull_optimisation(self, iloc, i_month, lactating_cow, dry_cow, month, remaining_months, once_aday_idx,
                           use_prod_price, use_pg, current_state, current_feed, use_feed_cost, cum_feed):
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
        assert cum_feed.shape == (1,)

        def cull_prod(ncull):
            rm_dry = np.min(np.concatenate((dry_cow, ncull)))
            new_dry_fract = dry_cow - rm_dry
            new_lact_fract = lactating_cow - (ncull - rm_dry)
            future_product, supplement_cost, deficit_feed = self._calc_cost_production(
                i_month=i_month,
                remaining_months=remaining_months,
                lact_fraction=new_lact_fract,
                dry_fraction=new_dry_fract,
                once_aday_idx=once_aday_idx,
                use_prod_price=use_prod_price,
                use_pg=use_pg,
                current_state=current_state,
                current_feed=current_feed,
                use_feed_cost=use_feed_cost,
                current_cum_import=cum_feed,
                nsims=1)
            return (future_product[0] - supplement_cost[0]) * -1

        res = minimize_scalar(cull_prod, bounds=(0, max_cull))

        cull = res.x
        assert res.success, f'optimisation failed {res.message}'
        rm_dry = np.min(np.concatenate((dry_cow, [cull])))
        new_dry_fract = dry_cow - rm_dry
        new_lact_fract = lactating_cow - (cull - rm_dry)
        return iloc, new_lact_fract[0], new_dry_fract[0]

    def _dryoff_opt_mp(self, kwargs):
        return self._dryoff_optmisation(**kwargs)

    def _dryoff_optmisation(self, iloc, i_month, lactating_cow, dry_cow, remaining_months, once_aday_idx,
                            use_prod_price, use_pg, current_state, current_feed, use_feed_cost,
                            cum_feed):
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
        assert cum_feed.shape == (1,)

        def dryoff_prod(ndry, rt_both=False):
            new_dry_fract = dry_cow + ndry
            new_lact_fract = lactating_cow - ndry
            future_product, supplement_cost, deficit_feed = self._calc_cost_production(
                i_month=i_month,
                remaining_months=remaining_months,
                lact_fraction=new_lact_fract,
                dry_fraction=new_dry_fract,
                once_aday_idx=once_aday_idx,
                use_prod_price=use_prod_price,
                use_pg=use_pg,
                current_state=current_state,
                current_feed=current_feed,
                use_feed_cost=use_feed_cost,
                current_cum_import=cum_feed,
                nsims=1)
            if rt_both:
                return future_product[0], supplement_cost[0]
            return (future_product[0] - supplement_cost[0]) * -1

        res = minimize_scalar(dryoff_prod, bounds=(0, max_dry))
        dry = res.x
        new_dry_fract = dry_cow + dry
        new_lact_fract = lactating_cow - dry
        return iloc, new_lact_fract[0], new_dry_fract[0]

    def _calc_cost_production(self, i_month, remaining_months, lact_fraction, dry_fraction, once_aday_idx,
                              use_prod_price,
                              use_pg, current_state, current_feed, use_feed_cost, nsims,
                              current_cum_import,
                              inc_scarcity=True):
        current_cum_import = current_cum_import.copy()
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

        assert geclose(deficit_feed, 0).all()

        supplement_needed = deficit_feed / self.supplemental_efficiency

        if inc_scarcity:
            scarcity_cost = self.calc_feed_scarcity_cost(
                i_month, start_cum_ann_feed_import=current_cum_import,
                new_feed=supplement_needed,
                use_sup_feed_cost=use_feed_cost,
                nsims=nsims)
        else:
            scarcity_cost = np.zeros((len(remaining_months), nsims))

        supplement_cost = (supplement_needed * use_feed_cost
                           + self.sup_feedout_cost * supplement_needed
                           + scarcity_cost).sum(axis=0)
        return future_product, supplement_cost, deficit_feed.sum(axis=0)

    def get_annual_feed(self):
        """
        get the annual feed needed for each farm

        :return: np.ndarray shape (nsims,)
        """

        if self._annual_feed_needed is not None:
            return self._annual_feed_needed

        remaining_months = np.concatenate([np.repeat(m, month_len[m])
                                           for m in [(self.month_reset - 1 + i) % 12 + 1 for i in range(12)]])
        future_product, supplement_cost, annual_feed = self._calc_cost_production(
            i_month=0,
            remaining_months=remaining_months,
            lact_fraction=np.ones(self.nsims),
            dry_fraction=np.zeros(self.nsims),
            once_aday_idx=np.zeros(self.nsims).astype(bool),
            use_prod_price=np.zeros((len(remaining_months), self.nsims)),
            use_pg=np.zeros((len(remaining_months), self.nsims)),
            current_state=np.zeros(self.nsims),
            current_feed=np.zeros(self.nsims),
            use_feed_cost=np.zeros((len(remaining_months), self.nsims)),
            nsims=self.nsims,
            current_cum_import=np.zeros(self.nsims),
            inc_scarcity=False
        )
        assert allclose(annual_feed, annual_feed[0])
        self._annual_feed_needed = annual_feed[0]
        return annual_feed


class DairyModelWithSCScarcity(SimpleDairyModel):
    """
    a version with an s-curve scarcity model requiring additional parameters s, a, b, c
    """
    s, a, b, c = None, None, None, None
    alt_kwargs = ('peak_lact_cow_per_ha', 'cull_dry_step', 'ncore_opt', 'logging_level', 'opt_mode', 's', 'a', 'b', 'c',
                  'cull_levels', 'dryoff_levels')

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price, opt_mode='optimised',
                 monthly_input=True,
                 s=None, a=None, b=None, c=None, peak_lact_cow_per_ha=default_peak_cow, ncore_opt=1,
                 logging_level=logging.CRITICAL, cull_dry_step=None, cull_levels=None, dryoff_levels=None):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number float or np.ndarray shape (nsims,), MJ/ha
        :param imoney: initial money number float or np.ndarray shape (nsims,) $/ha
        :param sup_feed_cost: cost of supplementary feed $/MJ float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param product_price: income price $/kg product float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param opt_mode: one of:

        * 'optimised': use scipy.minimize_scalar to optimise the fraction of cows to cull dryoff at each decision point (cpu intensive, but most precise, low memory) cull_dry_step must be None
        * 'step': use a fixed step size for cull/dryoff decision (less cpu intensive, less precise, low memory) cull_dry_step must be set
        * 'coarse': pick the best of an a. priori. number of cull/dryoff decisions (low cpu, high memory, moderate precision) cull_dry_step must be None

        :param monthly_input: if True, monthly input, if False, daily input (365 days per year)
        :param s: scarcity scurve scale - the maximum value of the curve (if s=1, the maximum value is 1)
        :param a: scarcity scurve steepness - smoothing parameter as a increases the curve becomes steeper and the inflection point moves to the right
        :param b: scarcity scurve steepness about the inflection point
        :param c: scarcity scurve inflection point (if a=1, c is the x value at which y=0.5)
        :param peak_lact_cow_per_ha: peak lactating cows per ha float
        :param ncore_opt: number of cores to use for optimization of the cull/dryoff decision if ncore_opt==1, do not multiprocess, if ncore_opt > 1, use ncore_opt cores, if ncore_opt =None, use all available cores
        :param logging_level: logging level for the multiprocess component
        :param cull_dry_step: if None, use the optimized cull/dryoff decision, if float, use the cull/dryoff decision with a step size of cull_dry_step
        :param cull_levels: if opt_mode='coarse' the number of equally spaced cull steps to make, otherwise must be None
        :param dryoff_levels: if opt_mode='coarse' the number of equally spaced dryoff steps to make, otherwise must be None
        """
        assert all([e is not None for e in [s, a, b, c]]), 's, a, b, c must not be None'
        assert np.isfinite(np.array([s, a, b, c])).all(), 's, a, b, c must be finite'
        self.s, self.a, self.b, self.c = float(s), float(a), float(b), float(c)
        super().__init__(all_months=all_months,
                         istate=istate,
                         pg=pg,
                         ifeed=ifeed,
                         imoney=imoney,
                         sup_feed_cost=sup_feed_cost,
                         product_price=product_price,
                         opt_mode=opt_mode,
                         monthly_input=monthly_input,
                         peak_lact_cow_per_ha=peak_lact_cow_per_ha, ncore_opt=ncore_opt, logging_level=logging_level,
                         cull_dry_step=cull_dry_step,
                         cull_levels=cull_levels,
                         dryoff_levels=dryoff_levels)

    def plot_scurve(self, plt_dnz_fs=False):
        """
        plot the s-curve
        :param plt_dnz_fs: bool if true plot the dairy nz farm system boundaries.
        :return:
        """

        x = np.linspace(0, 100, 100)
        y = s_curve(x, s=self.s, a=self.a, b=self.b, c=self.c)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, y)
        if plt_dnz_fs:
            ax.fill_between([1, 10], [0, 0], [1, 1], transform=ax.get_xaxis_transform(), color='r', alpha=0.1,
                            label='DNZ farm system 2')
            ax.fill_between([10, 20], [0, 0], [1, 1], transform=ax.get_xaxis_transform(), color='r', alpha=0.2,
                            label='DNZ farm system 3')
            ax.fill_between([20, 30], [0, 0], [1, 1], transform=ax.get_xaxis_transform(), color='r', alpha=0.3,
                            label='DNZ farm system 4')
            ax.fill_between([30, 100], [0, 0], [1, 1], transform=ax.get_xaxis_transform(), color='r', alpha=0.4,
                            label='DNZ farm systen 5')
        ax.legend()
        ax.set_xlabel('Feed import (% of annual full production demand)')
        ax.set_ylabel('Additional Feed scarcity cost (multiplier of feed cost)')
        ax.set_title('Feed scarcity cost as a function of feed import')
        fig.tight_layout()
        return fig, ax

    def calc_feed_scarcity_cost(self, i_month, start_cum_ann_feed_import, new_feed, use_sup_feed_cost, nsims):
        """
        Calculate the feed scarcity cost based on the previous cumulative feed import and the new feed import

        :param start_cum_ann_feed_import: previous cumulative feed import (one per sim)
        :param new_feed: new feed import (ndays, nsims)
        :return: feed scarcity cost ($ for the time period) (NOT $/MJ !) np.ndarray shape (ndays, nsims)
        """
        assert start_cum_ann_feed_import.shape == new_feed.shape[1:] == (
            nsims,), f'{start_cum_ann_feed_import.shape=} {new_feed.shape=}'

        start_cum_ann_feed_import = np.nanmax(np.concatenate((start_cum_ann_feed_import[np.newaxis],
                                                              np.zeros((1, nsims))), axis=0),
                                              axis=0)
        cum_feed_per = (start_cum_ann_feed_import + new_feed.cumsum(axis=0)) / self.get_annual_feed() * 100
        assert geclose(cum_feed_per.min(), 0), f'{cum_feed_per.min()}'
        assert leclose(cum_feed_per.max(), 105), f'{cum_feed_per.max()}'  # 105 to allow for rounding errors
        cum_feed_per[cum_feed_per > 100] = 100
        cost_per_mj = s_curve(
            cum_feed_per, s=self.s, a=self.a, b=self.b,
            c=self.c) * use_sup_feed_cost

        return cost_per_mj * new_feed


def s_curve(x, s, a, b, c):
    """
    S-curve function
    :param x: x value
    :param s: scale - the maximum value of the curve (if s=1, the maximum value is 1)
    :param a: steepness - smoothing parameter as a increases the curve becomes steeper and the inflection point moves to the right
    :param b: steepness about the inflection point
    :param c: inflection point (if a=1, c is the x value at which y=0.5)
    :return:
    """
    return s * (1 / (1 + np.exp(-b * (x - c)))) ** a
