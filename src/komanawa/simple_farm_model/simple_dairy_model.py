"""
created matt_dumont 
on: 6/15/24
"""
from komanawa.simple_farm_model.base_simple_farm_model import BaseSimpleFarmModel, month_len
import numpy as np
import pandas as pd
from pathlib import Path

mj_per_kg_dm = 11  # MJ ME /kg DM
monly_ms_prod = {  # kgMS per lactating cow per day # todo check with andrew
    1: 57.47,
    2: 49.10,
    3: 50.13,
    4: 45.89,
    5: 45.42,
    6: 0,  # todo andrew?
    7: 55.48,
    8: 56.92,
    9: 64.27,
    10: 68.17,
    11: 61.57,
    12: 60.67
}
daily_ms_prod = {m: v / month_len[m] for m, v in monly_ms_prod.items()}


class SimpleDairyModel(BaseSimpleFarmModel):
    precision = 2  # round to the nearest 0.01 cows
    states = {
        # i value: (2 a day, unmodified cow buckets)
        0: (True, True),  # base state day, not modified cow buckets
        1: (False, True),  # 1 a day, not modified cow buckets
        2: (True, False),  # 2 a day, modified cow buckets
        3: (False, False),  # 1 a day, modified cow buckets
    }
    mean_sup_cost = None  # set internally
    mean_pg = None  # set internally

    ndays_feed_import = 7  # when needing to import feed, import 7 days worth of feed (from 7 days supplemental needed)
    state_change_days = [1, 15]  # allow for a change in farm state on day 1 and 15 of each month
    month_reset = 7  # trigger farm reset on day 1 in August,
    homegrown_efficiency = 0.8
    supplemental_efficiency = 0.9
    sup_feedout_cost = 120 / 1000 / mj_per_kg_dm  # 120/ton DM --> convert to ME, MJ
    homegrown_store_efficiency = 1
    homegrown_storage_cost = 175 / 1000 / mj_per_kg_dm  # 175/ton/DM --> convert to ME, MJ
    one_a_daymilk_production_fraction = 1 - 0.13  # 13% less milk production on 1-a-day
    _start_lactating_cow_fraction = 0.8  # 80% of cows are lactating on day 1 # todo check with andrew, is there a natural change per month or only states...
    _start_dry_cow_fraction = 0  # 0% of cows are dry on day 1 # todo check with andrew
    _start_replacement_fraction = 0.2  # 20% of cows are replacements on day 1 # todo check with andrew
    peak_cow_per_ha = 2.5  # peak cow per ha # todo check with andrew (do we want to vary this?, how is it working)
    kgms_per_lac_cow = daily_ms_prod
    feed_per_cow_lactating = 1  # todo andrew? do we need to vary this by month?
    feed_per_cow_dry = 1  # todo andrew? do we need to vary this by month?
    feed_per_cow_replacement = 1  # todo andrew? do we need to vary this by month?
    feed_fraction_1aday = 0.87  # 13% less feed required on 1-a-day # todo dummy number andrew
    feed_store_trigger = 100  # trigger feed import if feed store is less than 100 MJ # todo dummy, andrew?
    marginal_benefit_from_true = True  # calculate marginal benefit from true prices, if False, use average price
    marginal_cost_from_true = True  # calculate marginal benefit from true prices, if False, use average price

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price, monthly_input=True):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number float or np.ndarray shape (nsims,)
        :param imoney: initial money number float or np.ndarray shape (nsims,)
        :param sup_feed_cost: cost of supplementary feed $/kgDM float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param product_price: income price $/kg product float or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param monthly_input: if True, monthly input, if False, daily input (365 days per year)
        """
        # add more input/outputs
        self.obj_dict = self.obj_dict.copy()
        self.obj_dict['lactating_cow_fraction'] = 'out_lactating_cow_fraction'
        self.obj_dict['dry_cow_fraction'] = 'out_dry_cow_fraction'
        self.obj_dict['replacement_fraction'] = 'out_replacement_fraction'
        self.obj_dict['feed_lactating'] = 'out_feed_lactating'
        self.obj_dict['feed_dry'] = 'out_feed_dry'
        self.obj_dict['feed_replacement'] = 'out_feed_replacement'
        self.obj_dict['out_cur_fut_prod'] = 'out_cur_fut_prod'
        self.obj_dict['out_alt_fut_prod'] = 'out_alt_fut_prod'
        self.obj_dict['out_marginal_cost'] = 'out_marginal_cost'
        self.obj_dict['out_alt_fut_feed_def'] = 'out_alt_fut_feed_def'
        self.obj_dict['out_cur_fut_feed_def'] = 'out_cur_fut_feed_def'
        self.obj_dict['marginal_benefit'] = 'out_marginal_benefit'

        self.long_name_dict = self.long_name_dict.copy()
        self.long_name_dict['lactating_cow_fraction'] = 'Lactating Cow Fraction (fraction, 0-1)'
        self.long_name_dict['dry_cow_fraction'] = 'Dry Cow Fraction (fraction, 0-1)'
        self.long_name_dict['replacement_fraction'] = 'Replacement Cow Fraction (fraction, 0-1)'
        self.long_name_dict['feed_lactating'] = 'Feed Lactating required on a given day (MJ)'
        self.long_name_dict['feed_dry'] = 'Feed Dry required on a given day (MJ)'
        self.long_name_dict['feed_replacement'] = 'Feed Replacement required on a given day (MJ)'
        self.long_name_dict['out_cur_fut_prod'] = 'Current Future Scenario Production for MC/MB ($)'
        self.long_name_dict['out_alt_fut_prod'] = 'Alternate Future Scenario Production for MC/MB ($)'
        self.long_name_dict['out_marginal_cost'] = 'Marginal Cost/Benefit for MC/MB ($)'
        self.long_name_dict['out_alt_fut_feed_def'] = 'Alternate Future Feed Deficit for MC/MB (MJ)'
        self.long_name_dict['out_cur_fut_feed_def'] = 'Current Future Feed Deficit for MC/MB (MJ)'
        self.long_name_dict['marginal_benefit'] = 'Marginal Benefit for MC/MB ($)'

        super().__init__(all_months=all_months, istate=istate, pg=pg,
                         ifeed=ifeed, imoney=imoney, sup_feed_cost=sup_feed_cost, product_price=product_price,
                         monthly_input=monthly_input)

        # identify which states are 1 a day milking
        self.milk_1_aday_states = [i for i, (a_day_2, _) in self.states.items() if not a_day_2]

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

        self.set_mean_pg(pg_dict=None, pg_raw=self.pg, months=all_months)
        self.set_mean_sup_cost(sup_dict=None, sup_cost_raw=self.sup_feedout_cost, months=all_months)

    def marginal_benefitset_mean_pg(self, pg_dict, pg_raw=None, months=None):
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
        assert isinstance(pg_dict, dict), f'pg must be dict, got {type(pg_dict)}'
        assert set(pg_dict.keys()) == set(range(1, 13)), f'pg keys must be 1-12, got {pg_dict.keys()}'

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
            assert months.shape == (self.nsims,), f'months must be shape {self.nsims=}, got {months.shape}'
            pg_dict = {}
            for m in range(1, 13):
                idx = months == m
                pg_dict[m] = pg_raw[idx].mean(axis=0)
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
            assert months.shape == (self.nsims,), f'months must be shape {self.nsims=}, got {months.shape}'
            sup_dict = {}
            for m in range(1, 13):
                idx = months == m
                sup_dict[m] = sup_cost_raw[idx].mean(axis=0)
        assert isinstance(sup_dict, dict), f'sup must be dict, got {type(sup_dict)}'
        assert set(sup_dict.keys()) == set(range(1, 13)), f'sup keys must be 1-12, got {sup_dict.keys()}'
        self.mean_sup_cost = sup_dict

    def convert_pg_to_me(self, pg, current_state):
        assert isinstance(pg, np.ndarray), f'pg must be np.ndarray, got {type(pg)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        return mj_per_kg_dm * pg

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        idxs = np.in1d(current_state, self.milk_1_aday_states)
        feed_lactating = self.feed_per_cow_lactating * self.lactating_cow_fraction
        feed_lactating[idxs] *= self.feed_fraction_1aday
        self.out_feed_lactating[i_month] = feed_lactating

        feed_dry = self.feed_per_cow_dry * self.dry_cow_fraction
        self.out_feed_dry[i_month] = feed_dry

        feed_replacement = self.feed_per_cow_replacement * self.replacement_fraction
        self.out_feed_replacement[i_month] = feed_replacement

        return feed_lactating + feed_dry + feed_replacement

    def calculate_production(self, i_month, month, current_state):
        max_production = (self.kgms_per_lac_cow[month]
                          * (self.peak_cow_per_ha * self.lactating_cow_fraction).round(self.precision))
        idxs = np.in1d(current_state, self.milk_1_aday_states)
        max_production[idxs] *= self.one_a_daymilk_production_fraction
        return max_production

    def reset_state(self, i_month):
        out = np.zeros(self.nsims)
        assert out.shape == (self.nsims,), f'out must be shape {self.nsims=}, got {out.shape}'

        # reset cow buckets
        self.lactating_cow_fraction = np.zeros(self.nsims) + self._start_lactating_cow_fraction
        self.dry_cow_fraction = np.zeros(self.nsims) + self._start_dry_cow_fraction
        self.replacement_fraction = np.zeros(self.nsims) + self._start_replacement_fraction

        return out  # all back to once a day milking

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

        # import feed where below threshold, import ndays worth of feed # todo document for andrew
        need_feed = current_feed < self.feed_store_trigger
        past_sup_use = self.sup_feed_needed[(i_month - self.ndays_feed_import + 1):i_month + 1].sum(axis=0)
        feed_imported = np.zeros(self.nsims)
        feed_imported[need_feed] = past_sup_use[need_feed]
        sup_feed_cost = feed_imported * self.sup_feed_cost[i_month]
        next_state = np.zeros(self.nsims) + current_state

        if self.all_days[i_month] in self.state_change_days:
            # allow state change
            # todo invoke feed store limit.
            marginal_cost, marginal_benefit = self.calc_marginal_cost_benefit(i_month, month, current_state)
            idx = marginal_cost > marginal_benefit
            reduced_states = self.calc_reduce_state(i_month, month, current_state, reduce_idx=idx)
            assert (reduced_states[~idx] == current_state[~idx]).all(), f'states changed when it should not have'
            next_state[idx] = reduced_states[idx]

        return next_state, feed_imported, sup_feed_cost

    def supplemental_action_last(self, i_month, month, day, current_state, current_feed,
                                 current_money):
        """
        supplemental actions that might be needed for sub models at the end of each time step.  This is a placeholder to allow rapid development without modifying the base class. it is absolutely reasonable to leave as is.
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
        return current_state, current_feed, current_money

    def calc_marginal_cost_benefit(self, i_month, month, current_state):
        current_year = self.all_year[i_month]
        stop_idx = np.argmax(self.all_year == current_year + 1)

        remaining_months = self.all_months[i_month:stop_idx]

        # todo may need to adjust for fewer cows if it's time varient beyond state..
        current_future_product = (
                np.array([self.kgms_per_lac_cow[m] for m in remaining_months])[:, np.newaxis]
                * (self.peak_cow_per_ha * self.lactating_cow_fraction[np.newaxis]).round(self.precision)
        )

        current_1day_idxs = np.in1d(current_state, self.milk_1_aday_states)
        current_future_product[:, current_1day_idxs] *= self.one_a_daymilk_production_fraction
        # convert max production to $
        if self.marginal_cost_from_true:
            current_future_product = current_future_product * self.product_price[i_month:stop_idx]
        else:
            current_future_product = current_future_product * self.product_price.mean()
        current_future_product = current_future_product.sum(axis=0)
        self.out_cur_fut_prod[i_month] = current_future_product

        (next_state, next_fraction_lact,
         next_fraction_dry, next_fraction_replacement) = self._calc_next_state_quant(current_state)

        alternate_production = (np.array([self.kgms_per_lac_cow[m] for m in remaining_months])[:, np.newaxis]
                                * (self.peak_cow_per_ha * next_fraction_lact[np.newaxis]).round(self.precision))
        alt_1day_idxs = np.in1d(next_state, self.milk_1_aday_states)
        alternate_production[:, alt_1day_idxs] *= self.one_a_daymilk_production_fraction
        # convert max production to $
        if self.marginal_cost_from_true:
            alternate_production = alternate_production * self.product_price[i_month:stop_idx]
        else:
            alternate_production = alternate_production * self.product_price.mean()
        alternate_production = alternate_production.sum(axis=0)
        self.out_alt_fut_prod[i_month] = alternate_production

        marginal_cost = current_future_product - alternate_production
        self.out_marginal_cost[i_month] = marginal_cost

        # calculate marginal benefit
        if self.marginal_benefit_from_true:
            in_pg = self.pg[i_month:stop_idx]

        else:
            in_pg = np.zeros(len(remaining_months), self.nsims)
            for i in range(month, 13):
                idx = remaining_months == i
                in_pg[idx] = self.mean_pg[i]

        in_pg = in_pg.copy()

        # keynote assume all feed is eaten on pasture (no storage/storage efficiency)
        current_home_production = self.convert_pg_to_me(in_pg, current_state) * self.homegrown_store_efficiency
        alt_home_production = self.convert_pg_to_me(in_pg, next_state) * self.homegrown_store_efficiency

        # calculate the amount of feed required for the current/alternate state
        current_feed_lactating = self.feed_per_cow_lactating * self.lactating_cow_fraction
        current_feed_lactating[current_1day_idxs] *= self.feed_fraction_1aday
        current_feed_dry = self.feed_per_cow_dry * self.dry_cow_fraction
        current_feed_replacement = self.feed_per_cow_replacement * self.replacement_fraction
        current_feed = current_feed_lactating + current_feed_dry + current_feed_replacement
        current_feed_deficit = current_feed - current_home_production
        current_feed_deficit *= self.supplemental_efficiency
        self.out_cur_fut_feed_def[i_month] = current_feed_deficit

        alt_feed_lactating = self.feed_per_cow_lactating * next_fraction_lact
        alt_feed_lactating[alt_1day_idxs] *= self.feed_fraction_1aday
        alt_feed_dry = self.feed_per_cow_dry * next_fraction_dry
        alt_feed_replacement = self.feed_per_cow_replacement * next_fraction_replacement
        alt_feed = alt_feed_lactating + alt_feed_dry + alt_feed_replacement
        alt_feed_deficit = alt_feed - alt_home_production
        alt_feed_deficit *= self.supplemental_efficiency
        self.out_alt_fut_feed_def[i_month] = alt_feed_deficit

        # calc usefeed_cost
        if self.marginal_benefit_from_true:
            use_feed_cost = self.sup_feed_cost[i_month:stop_idx]
        else:
            use_feed_cost = np.zeros(len(remaining_months), self.nsims)
            for i in range(month, 13):
                idx = remaining_months == i
                use_feed_cost[idx] = self.mean_sup_cost[i]

        use_feed_cost = use_feed_cost.copy()

        marginal_benefit = (
                (current_feed_deficit * self.sup_feedout_cost + current_feed_deficit * use_feed_cost)
                - (alt_feed_deficit * self.sup_feedout_cost + alt_feed_deficit * use_feed_cost)
        )
        self.out_marginal_benefit[i_month] = marginal_benefit

        return marginal_cost, marginal_benefit

    def calc_reduce_state(self, i_month, month, current_state, reduce_idx):
        """
        reduce state based on reduce_idx

        :param i_month: day of simulation
        :param month: month of simulation
        :param current_state: the current state for all farms
        :param reduce_idx: those farms that should reduce state
        :return:
        """
        out_next_state = np.zeros(self.nsims) + current_state

        (next_state, next_fraction_lact, next_fraction_dry,
         next_fraction_replacement) = self._calc_next_state_quant(current_state)
        out_next_state[reduce_idx] = next_state[reduce_idx]
        self.lactating_cow_fraction[reduce_idx] = next_fraction_lact[reduce_idx]
        self.dry_cow_fraction[reduce_idx] = next_fraction_dry[reduce_idx]
        self.replacement_fraction[reduce_idx] = next_fraction_replacement[reduce_idx]
        return out_next_state

    def _calc_next_state_quant(self, current_state):
        # todo some threshold on the amount of feed imported in the last fornight, etc.
        # todo calculate the next quant state down, does this rely on the month? preivous avaliblity, past feed import, etc...
        # options = 1 a day, reduce cattle numbers, dry off cattle, etc.
        raise NotImplementedError
        return next_state, next_fraction_lact, next_fraction_dry, next_fraction_replacement
