"""
created matt_dumont 
on: 26/06/23

This module contains the BaseSimpleFarmModel class and its child class DummySimpleFarm. These classes are used to
simulate a simple farm economic model.

The BaseSimpleFarmModel class is an abstract base class that provides the basic structure and methods for the farm
economic model. It includes methods for running the model, saving results to a netCDF file, reading in a model from a
netCDF file, saving results to CSV files, and plotting results.

The DummySimpleFarm class is a child class of BaseSimpleFarmModel. It is a placeholder class that needs to be
implemented with specific farm model logic.

The module also includes a helper function get_colors() for generating a list of colors for plotting, and a dictionary
month_len that provides the number of days in each month.
"""
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import netCDF4 as nc
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class BaseSimpleFarmModel(object):
    """
    Abstract base class for a simple farm economic model.

    This class provides the basic structure and methods for the farm economic model. It includes methods for running
    the model, saving results to a netCDF file, reading in a model from a netCDF file, saving results to CSV files,
    and plotting results.

    The class needs to be subclassed and the following methods need to be implemented in the subclass:

    - calculate_feed_needed()
    - calculate_production()
    - calculate_next_state()
    - calculate_sup_feed()
    - calculate_running_cost()
    - reset_state()

    Class Attributes:

       - :cvar long_name_dict: (dict) Dictionary of attribute name to attribute long name (unit) for display and metadata.
       - :cvar obj_dict: (dict) Dictionary of attribute name to object name. used to

    Subclass Attributes that must be set in the subclass:

        :cvar states: (dict): Dictionary of state numbers to state descriptions.
        :cvar month_reset: (int or None): Month number when the farm state is reset. If None then the farm state is not reset.

    Instance Attributes:

        - Descriptor attributes
            - :ivar nsims: (int): Number of simulations.
            - :ivar time_len: (int): Number of time steps in the model.
            - :ivar model_shape: (tuple): Shape of the model arrays (time_len + 1, nsims).
            - :ivar _model_input_shape: (tuple): Shape of the model input arrays (time_len, nsims).
            - :ivar inpath: (Path or None) Path to the input file, if the model was read in from a file.
            - :ivar all_days: (np.ndarray): Array of day numbers for each time step in the model.
            - :ivar all_months: (np.ndarray): Array of month numbers for each time step in the model.
            - :ivar all_year: (np.ndarray): Array of year numbers for each time step in the model.
            - :ivar _run: (bool): Flag indicating if the model has been run.

        - Model input attributes (all transition to model_shape (time, nsims))
            - :ivar interest_rate: (np.ndarray): Array of interest rates for each time step in the model.
            - :ivar pg: (np.ndarray): Array of pasture growth amounts for each time step in the model.
            - :ivar product_price: (np.ndarray): Array of product prices for each time step in the model.
            - :ivar sup_feed_cost: (np.ndarray): Array of supplementary feed costs for each time step in the model.

        - Model calculation attributes
            - :ivar model_debt_service: (np.ndarray): Array of debt service amounts for each time step in the model.
            - :ivar model_feed: (np.ndarray): Array of feed amounts for each time step in the model.
            - :ivar model_feed_cost: (np.ndarray): Array of feed cost amounts for each time step in the model.
            - :ivar model_feed_demand: (np.ndarray): Array of feed demand amounts for each time step in the model.
            - :ivar model_feed_imported: (np.ndarray): Array of imported feed amounts for each time step in the model.
            - :ivar model_money: (np.ndarray): Array of money amounts for each time step in the model.
            - :ivar model_prod: (np.ndarray): Array of product amounts for each time step in the model.
            - :ivar model_prod_money: (np.ndarray): Array of product money amounts for each time step in the model.
            - :ivar model_running_cost: (np.ndarray): Array of running cost amounts for each time step in the model.
            - :ivar model_state: (np.ndarray): Array of state numbers for each time step in the model.

    Process:

        The run_model method in the BaseSimpleFarmModel class is responsible for running the farm economic model. Here's a step-by-step description of the process:

        #. The method first checks if the model has already been run. If it has, it raises an assertion error.
        #. It then enters a loop that iterates over each month in the model's timeline. For each month, it performs the following steps:
            #. It sets the start of day values for current money, current feed, and current state.
            #. It calculates the pasture growth and adds it to the current feed.
            #. It calculates the feed needed for the cattle and subtracts it from the current feed. The feed demand is also stored.
            #. It calculates the product produced and stores it.
            #. It calculates the profit from selling the product and adds it to the current money.
            #. It calculates the supplementary feed needed, adds it to the current feed, and stores it.
            #. It calculates the cost of the supplementary feed and subtracts it from the current money. The feed cost is also stored.
            #. It calculates the running costs and subtracts them from the current money. The running costs are also stored.
            #. It calculates the debt servicing costs and subtracts them from the current money. The debt servicing costs are also stored.
            #. It calculates the next state of the farm. If it's the start of a new year, it resets the state.
            #. Finally, it sets the key values for the current state, feed, and money.
        #. After the loop, it sets the _run attribute to True, indicating that the model has been run.

        This method essentially simulates the operation of the farm over time, taking into account various factors such as feed demand, product production, costs, and state transitions.

    """
    # create dictionary of attribute name to attribute long name (unit)
    long_name_dict = {
        'state': 'farm state (none)',
        'feed': 'feed on farm (kgDM/ha)',
        'money': 'profit/loss (NZD)',
        'feed_demand': 'feed demand (kgDM/ha)',
        'prod': 'product produced (kg/ha)',
        'prod_money': 'profit from product (NZD)',
        'feed_imported': 'supplemental feed imported (kgDM/ha)',
        'feed_cost': 'cost of feed imported (NZD)',
        'running_cost': 'general farm running cost (ex. feed beyond expected feed) (NZD)',
        'debt_servicing': 'debt servicing (NZD)',
        'pg': 'pasture growth (kgDM/ha)',
        'product_price': 'product price (NZD/kg)',
        'feed_price': 'cost of feed imported (NZD)',
        'interest_rate': 'interest rate (%/yr)'

    }
    # create dictionary of attribute name to object name
    obj_dict = {
        'state': 'model_state',
        'feed': 'model_feed',
        'money': 'model_money',
        'feed_demand': 'model_feed_demand',
        'prod': 'model_prod',
        'prod_money': 'model_prod_money',
        'feed_imported': 'model_feed_imported',
        'feed_cost': 'model_feed_cost',
        'running_cost': 'model_running_cost',
        'debt_servicing': 'model_debt_service',
        'pg': 'pg',
        'product_price': 'product_price',
        'feed_price': 'sup_feed_cost',
        'interest_rate': 'interest_rate'
    }
    inpath = None

    states = None  # dummy value, must be set in child class
    month_reset = None  # dummy value, must be set in child class
    all_days = None
    all_months = None
    all_year = None
    interest_rate = None
    model_debt_service = None
    model_feed = None
    model_feed_cost = None
    model_feed_demand = None
    model_feed_imported = None
    model_money = None
    model_prod = None
    model_prod_money = None
    model_running_cost = None
    model_shape = None
    model_state = None
    nsims = None
    pg = None
    product_price = None
    sup_feed_cost = None
    time_len = None

    def __init__(self, all_months, istate, pg, ifeed, imoney, sup_feed_cost, product_price, interest_rate,
                 monthly_input=True):
        """

        :param all_months: integer months, defines mon_len and time_len
        :param istate: initial state number or np.ndarray shape (nsims,) defines number of simulations
        :param pg: pasture growth kgDM/ha/day np.ndarray shape (mon_len,) or (mon_len, nsims)
        :param ifeed: initial feed number or np.ndarray shape (nsims,)
        :param imoney: initial money number or np.ndarray shape (nsims,)
        :param sup_feed_cost: cost of supplementary feed $/kgDM or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param product_price: income price $/kg product or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param interest_rate: interest rate %/year or np.ndarray shape (nsims,) or (mon_len, nsims)
        :param monthly_input: if True, monthly input, if False, daily input
        """
        # define model shape
        assert set(all_months).issubset(set(range(1, 13))), f'months must be in range 1-12'
        all_month_org = deepcopy(all_months)
        if monthly_input:
            org_month_len = len(all_months)
            self.time_len = np.sum([month_len[m] for m in all_months])
        else:
            org_month_len = None  # should not ever access
            self.time_len = len(all_months)
        self.nsims = len(istate)
        self.model_shape = (self.time_len + 1, self.nsims)
        self._model_input_shape = (self.time_len, self.nsims)
        self._run = False

        assert all(len(e) == self.nsims for e in (ifeed, imoney))
        if monthly_input:
            all_months = np.concatenate([np.repeat(m, month_len[m]) for m in all_month_org])
            all_days = np.concatenate([np.arange(1, month_len[m] + 1) for m in all_month_org])
        else:
            all_days = pd.Series(all_months == np.concatenate(([all_months[0]], all_months[:-1])))
            all_days[0] = False
            all_days = all_days.cumsum() - (all_days.cumsum().where(~all_days).ffill().fillna(0).astype(int))
            all_days = all_days.values + 1
        all_year = np.cumsum((all_days == 1) & (all_months == 1))
        ifeed = np.atleast_1d(ifeed)
        imoney = np.atleast_1d(imoney)

        # handle pg input options
        pg = np.atleast_1d(pg)
        if pg.ndim == 1:
            if monthly_input:
                assert len(pg) == org_month_len, f'pg must be length time: {org_month_len=}, got: {len(pg)=}'
                pg = np.concatenate([np.repeat(p, month_len[m]) for p, m in zip(pg, all_month_org)])
            else:
                assert len(pg) == self.time_len, f'pg must be length time: {self.time_len=}, got: {len(pg)=}'
            pg = np.repeat(pg[:, np.newaxis], self.nsims, axis=1)
        elif monthly_input:
            pg = np.concatenate([np.repeat(p[np.newaxis], month_len[m], axis=0) for p, m in zip(pg, all_month_org)])
        self.pg = np.concatenate([pg[[0]], pg], axis=0)
        assert self.pg.shape == self.model_shape, f'pg shape must be: {self.model_shape=}, got: {pg.shape=}'

        # handle sup_feed_cost input options
        self.sup_feed_cost = self._manage_input_shape('sup_feed_cost', sup_feed_cost, monthly_input,
                                                      org_month_len, all_month_org)

        # manage product price options
        self.product_price = self._manage_input_shape('product_price', product_price, monthly_input,
                                                      org_month_len, all_month_org)

        self.interest_rate = self._manage_input_shape('interest_rate', interest_rate, monthly_input,
                                                      org_month_len, all_month_org)  # todo propage through
        if np.nanmax(self.interest_rate) < 1:
            raise ValueError(f'interest_rate must be in %/year, got: {np.nanmax(self.interest_rate)=} '
                             f'which is less than 1 suggesting it is in fraction/year')

        assert set(istate).issubset(
            set(self.states.keys())), f'unknown istate: {set(istate)} must be one of {self.states}'

        # setup key model values
        self.model_state = np.full(self.model_shape, -1, dtype=int)
        self.model_feed = np.full(self.model_shape, np.nan)
        self.model_money = np.full(self.model_shape, np.nan)

        # setup output values
        self.model_feed_demand = np.full(self.model_shape, np.nan)
        self.model_prod = np.full(self.model_shape, np.nan)
        self.model_prod_money = np.full(self.model_shape, np.nan)
        self.model_feed_imported = np.full(self.model_shape, np.nan)
        self.model_feed_cost = np.full(self.model_shape, np.nan)
        self.model_running_cost = np.full(self.model_shape, np.nan)
        self.model_debt_service = np.full(self.model_shape, np.nan)

        # set initial values
        self.model_state[0, :] = istate
        self.model_feed[0, :] = ifeed
        self.model_money[0, :] = imoney
        self.all_months = np.concatenate([[0], all_months])
        self.all_days = np.concatenate([[0], all_days])
        self.all_year = np.concatenate([[0], all_year])

        # internal check of shapes
        for v in self.obj_dict.values():
            assert getattr(self, v).shape == self.model_shape, (f'bad shape for {v} {getattr(self, v).shape=} '
                                                                f'must be {self.model_shape=}')
        # ntime shapes
        for v in [self.all_months, self.all_days, self.all_year]:
            assert v.shape == (
                self.model_shape[0],), f'bad shape for {v} {v.shape=} must be {(self.model_shape[0],)}'

    def run_model(self):
        """
        run the model see docstring for process
        :return:
        """
        assert not self._run, f'model has already been run'
        for i_month in range(1, self.time_len + 1):
            month = self.all_months[i_month]
            day = self.all_days[i_month]

            # set start of day values
            current_money = deepcopy(self.model_money[i_month - 1, :])
            current_feed = deepcopy(self.model_feed[i_month - 1, :])
            current_state = deepcopy(self.model_state[i_month - 1, :])

            # pasture growth
            current_feed += self.pg[i_month, :]

            # feed cattle
            feed_needed = self.calculate_feed_needed(i_month, month, current_state)
            current_feed = current_feed - feed_needed
            self.model_feed_demand[i_month, :] = feed_needed

            # produce product
            produced_product = self.calculate_production(i_month, month, current_state)
            self.model_prod[i_month, :] = produced_product

            # sell product
            prod_money = produced_product * self.product_price[i_month, :]
            self.model_prod_money[i_month, :] = prod_money
            current_money += prod_money

            sup_feed = self.calculate_sup_feed(i_month, month, current_state)
            self.model_feed_imported[i_month, :] = sup_feed
            current_feed += sup_feed

            sup_feed_cost = sup_feed * self.sup_feed_cost[i_month, :]
            self.model_feed_cost[i_month, :] = sup_feed_cost
            current_money -= sup_feed_cost

            # add running_cost
            run_costs = self.calculate_running_cost(i_month, month, current_state)
            self.model_running_cost[i_month, :] = run_costs
            current_money -= run_costs

            # add debt servicing
            debt_servicing = self.model_money[i_month - 1, :] * (1 + self.interest_rate[i_month, :] / 100)
            self.model_debt_service[i_month, :] = debt_servicing
            current_money -= debt_servicing

            # todo debt service ratio servicing G:\Shared drives\Z20002SLM_SLMACC\farm_model\farm health index.xlsx
            #  over what period?, here it is cumulative for the model run
            # todo include (capital spend/depreciation,tax, mangment wage) or is this in the run costs (NOT IN RUN COSTS)
            # todo still need to save the data., add to attribute description
            # todo used for each year, typically from 1 aug from reset to reset.
            opt_surplus = self.model_prod_money - (self.model_feed_cost + self.model_running_cost)
            debt_service_ratio = (np.nansum(opt_surplus[:i_month, :], axis=0)
                                  / np.nansum(self.model_debt_service[:i_month + 1, :], axis=0))

            # calculate next state
            next_state = self.calculate_next_state(i_month, month, current_state)
            # new year? reset state
            if month == self.month_reset and day == 1:
                next_state = self.reset_state(i_month)

            # set key values
            self.model_state[i_month, :] = next_state
            self.model_feed[i_month, :] = current_feed
            self.model_money[i_month, :] = current_money

        self._run = True

    def _manage_input_shape(self, input_key, input_val, monthly_input, org_month_len, all_month_org):
        """
        manage the shape of input that can be a scalar, a vector or a matrix all end up as np.ndarray shape (time, nsims)

        :param input_key: input key (for error messages)
        :param input_val: input value (number or np.array)
        :param monthly_input: pass through variables
        :param org_month_len: pass through variables
        :param all_month_org:  pass through variables
        :return:
        """

        if pd.api.types.is_number(input_val):
            input_val = np.full(self._model_input_shape, input_val)
        elif input_val.ndim == 1:
            if monthly_input:
                assert len(input_val) == org_month_len, (f'{input_key} must be length time: {org_month_len=}, '
                                                         f'got: {len(input_val)=}')
                input_val = np.concatenate(
                    [np.repeat(e, month_len[m]) for e, m in zip(input_val, all_month_org)])
            else:
                assert len(input_val) == self.time_len, (f'{input_key} must be length time: {self.time_len=}, '
                                                         f'got: {len(input_val)=}')
            input_val = np.repeat(input_val[:, np.newaxis], self.nsims, axis=1)
        elif monthly_input:
            input_val = np.concatenate(
                [np.repeat(e[np.newaxis], month_len[m], axis=0) for e, m in zip(input_val, all_month_org)])
        out = np.concatenate([input_val[[0]], input_val], axis=0)
        assert out.shape == self.model_shape, (f'{input_key} shape must be: {self.model_shape=}, '
                                               f'got: {out.shape=}')
        return out

    def save_results_nc(self, outpath):
        """
        save results to netcdf file

        :param outpath: path to save file
        :return:
        """
        assert self._run, 'model must be run before saving results'
        assert isinstance(outpath, Path) or isinstance(outpath,
                                                       str), f'outpath must be Path or str, got: {type(outpath)}'
        outpath = Path(outpath)
        outpath.parent.mkdir(exist_ok=True, parents=True)

        with nc.Dataset(outpath, 'w') as ds:
            ds.createDimension('time', self.time_len + 1)
            ds.createDimension('nsims', self.nsims)

            ds.createVariable('sims', 'i4', ('nsims',))
            ds.variables['sims'][:] = range(self.nsims)
            ds.variables['sims'].setncattr('long_name', 'simulation number')

            ds.createVariable('month', 'i4', ('time',))
            ds.variables['month'][:] = self.all_months
            ds.variables['month'].setncattr('long_name', 'month of the year')

            ds.createVariable('year', 'i4', ('time',))
            ds.variables['year'].setncattr('long_name', 'year')
            ds.variables['year'][:] = self.all_year

            ds.createVariable('day', 'i4', ('time',))
            ds.variables['day'].setncattr('long_name', 'day of the month')
            ds.variables['day'][:] = self.all_days

            ds.createVariable('state', 'i4', ('time', 'nsims'))
            ds.variables['state'].setncatts({'long_name': 'farm state', 'units': 'none'})
            ds.variables['state'][:] = self.model_state

            ds.createVariable('feed', 'f8', ('time', 'nsims'))
            ds.variables['feed'].setncatts({'long_name': 'feed on farm', 'units': 'kgDM/ha'})
            ds.variables['feed'][:] = self.model_feed

            ds.createVariable('money', 'f8', ('time', 'nsims'))
            ds.variables['money'].setncatts({'long_name': 'profit/loss', 'units': 'NZD'})
            ds.variables['money'][:] = self.model_money

            ds.createVariable('feed_demand', 'f8', ('time', 'nsims'))
            ds.variables['feed_demand'].setncatts({'long_name': 'feed demand', 'units': 'kgDM/ha'})
            ds.variables['feed_demand'][:] = self.model_feed_demand

            ds.createVariable('prod', 'f8', ('time', 'nsims'))
            ds.variables['prod'].setncatts({'long_name': 'product produced', 'units': 'kg/ha'})
            ds.variables['prod'][:] = self.model_prod

            ds.createVariable('prod_money', 'f8', ('time', 'nsims'))
            ds.variables['prod_money'].setncatts({'long_name': 'profit from product', 'units': 'NZD'})
            ds.variables['prod_money'][:] = self.model_prod_money

            ds.createVariable('feed_imported', 'f8', ('time', 'nsims'))
            ds.variables['feed_imported'].setncatts({'long_name': 'supplemental feed imported', 'units': 'kgDM/ha'})
            ds.variables['feed_imported'][:] = self.model_feed_imported

            ds.createVariable('feed_cost', 'f8', ('time', 'nsims'))
            ds.variables['feed_cost'].setncatts({'long_name': 'cost of feed imported', 'units': 'NZD'})
            ds.variables['feed_cost'][:] = self.model_feed_cost

            ds.createVariable('running_cost', 'f8', ('time', 'nsims'))
            ds.variables['running_cost'].setncatts(
                {'long_name': 'general farm running cost (ex. feed beyond expected feed)', 'units': 'NZD'})
            ds.variables['running_cost'][:] = self.model_running_cost

            ds.createVariable('debt_servicing', 'f8', ('time', 'nsims'))
            ds.variables['debt_servicing'].setncatts(
                {'long_name': 'debt servicing', 'units': 'NZD'})
            ds.variables['debt_servicing'][:] = self.model_debt_service

            ds.createVariable('pg', 'f8', ('time', 'nsims'))
            ds.variables['pg'].setncatts({'long_name': 'pasture growth', 'units': 'kgDM/ha'})
            ds.variables['pg'][:] = self.pg

            ds.createVariable('product_price', 'f8', ('time', 'nsims'))
            ds.variables['product_price'].setncatts({'long_name': 'product price', 'units': 'NZD/kg'})
            ds.variables['product_price'][:] = self.product_price

            ds.createVariable('feed_price', 'f8', ('time', 'nsims'))
            ds.variables['feed_price'].setncatts({'long_name': 'feed price', 'units': 'NZD/kg'})
            ds.variables['feed_price'][:] = self.sup_feed_cost

            ds.createVariable('interest_rate', 'f8', ('time', 'nsims'))
            ds.variables['interest_rate'].setncatts({'long_name': 'interest rate', 'units': 'percent/yr'})
            ds.variables['interest_rate'][:] = self.interest_rate

    @classmethod
    def from_input_file(self, inpath):
        """
        read in a model from a netcdf file

        :param inpath:
        :return:
        """
        assert isinstance(inpath, Path) or isinstance(inpath,
                                                      str), f'inpath must be Path or str, got: {type(inpath)}'
        inpath = Path(inpath)
        if not inpath.exists():
            raise FileNotFoundError(f'file does not exist: {inpath}')

        with nc.Dataset(inpath, 'r') as ds:
            all_months = np.array(ds.variables['month'][:])
            pg = np.array(ds.variables['pg'][:])
            product_price = np.array(ds.variables['product_price'][:])
            model_state = np.array(ds.variables['state'][:])
            model_feed = np.array(ds.variables['feed'][:])
            model_money = np.array(ds.variables['money'][:])
            sup_feed_cost = np.array(ds.variables['feed_price'][:])
            interest_rate = np.array(ds.variables['interest_rate'][:])

            out = self(all_months=all_months[1:], istate=model_state[0], pg=pg[1:],
                       ifeed=model_feed[0], imoney=model_money[0], sup_feed_cost=sup_feed_cost[1:],
                       product_price=product_price[1:], interest_rate=interest_rate[1:], monthly_input=False)
            out.model_state = model_state
            out.model_feed = model_feed
            out.model_money = model_money

            out.model_feed_demand = np.array(ds.variables['feed_demand'][:])
            out.model_prod = np.array(ds.variables['prod'][:])
            out.model_prod_money = np.array(ds.variables['prod_money'][:])
            out.model_feed_imported = np.array(ds.variables['feed_imported'][:])
            out.model_feed_cost = np.array(ds.variables['feed_cost'][:])
            out.model_running_cost = np.array(ds.variables['running_cost'][:])
            out.model_debt_service = np.array(ds.variables['debt_servicing'][:])

        # set run status
        out._run = True
        out.inpath = inpath
        return out

    def save_results_csv(self, outdir, sims=None):
        """
        save results to csv in outdir with name sim_{sim}.csv

        :param outdir: output directory
        :param sims: sim numbers to save, if None, save all
        :return:
        """
        assert self._run, 'model must be run before saving results'
        assert isinstance(outdir, Path) or isinstance(outdir, str), f'outdir must be Path or str, got: {type(outdir)}'
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        if sims is None:
            sims = range(self.nsims)
        else:
            sims = np.atleast_1d(sims)

        for sim in sims:
            outdata = pd.DataFrame({
                'year': self.all_year,
                'month': self.all_months,
                'day': self.all_days,
                'state': self.model_state[:, sim],
                'feed': self.model_feed[:, sim],
                'money': self.model_money[:, sim],
                'feed_demand': self.model_feed_demand[:, sim],
                'prod': self.model_prod[:, sim],
                'prod_money': self.model_prod_money[:, sim],
                'feed_imported': self.model_feed_imported[:, sim],
                'feed_cost': self.model_feed_cost[:, sim],
                'running_cost': self.model_running_cost[:, sim],
                'debt_service': self.model_debt_service[:, sim],
                'pg': self.pg[:, sim],
                'product_price': self.product_price[:, sim],
                'sup_feed_cost': self.sup_feed_cost[:, sim],
                'interest_rate': self.interest_rate[:, sim]
            })
            outdata.to_csv(outdir.joinpath(f'sim_{sim}.csv'), index=False)

    def plot_results(self, *ys, x='time', sims=None, mult_as_lines=True, twin_axs=False, figsize=(10, 8),
                     start_year=2000, bins=20):
        """
        plot results

        :param ys: one or more of:

            - 'state' (state of the system)
            - 'feed' (feed on farm)
            - 'money' (money on farm)
            - 'feed_demand' (feed demand)
            - 'prod' (production)
            - 'prod_money' (production money)
            - 'feed_imported' (feed imported)
            - 'feed_cost' (feed cost)
            - 'running_cost' (running cost)
            - 'debt_service' (debt servicing)
            - 'pg' (pasture growth)
            - 'product_price' (product price)
            - 'sup_feed_cost' (supplementary feed cost)

        :param x: x axis to plot against 'time' or a ys variable
        :param sims: sim numbers to plot
        :param mult_as_lines: bool if True, plot multiple sims as lines, if False, plot multiple sims boxplot style pdf
        :param twin_axs: bool if True, plot each y on twin axis, if False, plot on subplots
        :param start_year: the year to start plotting from (default 2000) only affects the transition from year 1, year2, etc. to datetime
        :param bins: number of bins for mult_as_lines=False and x!='time'
        :return:
        """
        ys = np.atleast_1d(ys)
        assert set(ys).issubset(self.long_name_dict.keys()), (
            f'ys must be one or more of: {self.long_name_dict.keys()}, got'
            f'{set(ys) - set(self.long_name_dict.keys())}')
        assert x in self.long_name_dict.keys() or x == 'time', f'x must be "time" or one of: {self.long_name_dict.keys()}, got {x}'
        assert isinstance(mult_as_lines, bool), f'mult_as_lines must be bool, got {type(mult_as_lines)}'
        assert isinstance(twin_axs, bool), f'twin_axs must be bool, got {type(twin_axs)}'
        if twin_axs and len(ys) > 2:
            raise NotImplementedError('twin_axs only implemented for 1 or 2 ys, it gets super messy otherwise')

        if sims is None:
            sims = range(self.nsims)

        sims = np.atleast_1d(sims)
        assert set(sims).issubset(range(self.nsims)), f'sims must be in range({self.nsims}), got {sims}'

        if not mult_as_lines and len(sims) == 1:
            mult_as_lines = True
            warnings.warn('only one sim to plot, setting mult_as_lines=True')

        if len(sims) > 20 and mult_as_lines:
            warnings.warn('more than 20 sims to plot, colors will be reused')
        base_xtime = None
        if x == 'time':
            xlabel = 'time'
            base_xtime = np.array([datetime.date(
                year=start_year + yr, month=m, day=d
            ) for yr, m, d in zip(self.all_year[1:], self.all_months[1:], self.all_days[1:])])
            base_xtime = np.concatenate([[base_xtime[0] - datetime.timedelta(days=1)], base_xtime])
        else:
            xlabel = self.long_name_dict[x]

        # setup plots
        if twin_axs:
            axs = []
            linestyles = ['solid', 'dashed']
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=figsize)
            axs.append(ax)
            if len(ys) > 1:
                for y in ys[1:]:
                    axs.append(ax.twinx())
        else:
            fig, axs = plt.subplots(len(ys), 1, sharex=True, figsize=figsize)
            if isinstance(axs, plt.Axes):
                axs = [axs]
            linestyles = ['solid'] * len(ys)

        ycolors = get_colors(ys)

        # plot data
        for y, ax, ls, yc in zip(ys, axs, linestyles, ycolors):

            if mult_as_lines:
                sim_colors = get_colors(sims)
                for sim, c in zip(sims, sim_colors):
                    if x == 'time':
                        use_x = base_xtime
                        assert use_x is not None, 'base_xtime must be set'
                        idx = np.ones(use_x.shape).astype(bool)
                    else:
                        use_x = getattr(self, self.obj_dict[x])[:, sim]
                        idx = np.argsort(use_x)
                    if twin_axs:
                        label = f'sim: {sim}, {y}'
                    else:
                        label = f'sim {sim}'
                    ax.plot(use_x[idx], getattr(self, self.obj_dict[y])[:, sim][idx], label=label, ls=ls, c=c)
            else:
                usey = getattr(self, self.obj_dict[y])[:, sims]
                if x == 'time':
                    use_x = base_xtime
                    marker = None

                else:
                    marker = 'o'
                    all_x = getattr(self, self.obj_dict[x])[:, sims]
                    bin_limits = np.histogram_bin_edges(all_x.flatten(), bins=bins)
                    use_x = np.mean(np.stack([bin_limits[:-1], bin_limits[1:]]), axis=0)
                    tempy = [usey[(all_x >= bin_limits[i]) & (all_x < bin_limits[i + 1])] for i in
                             range(len(bin_limits) - 1)]
                    maxy = np.max([len(t) for t in tempy])
                    tempy = [np.pad(t, (0, maxy - len(t)), mode='constant', constant_values=np.nan) for t in tempy]
                    usey = np.stack(tempy, axis=0)

                ax.plot(use_x, np.nanmedian(usey, axis=1), label='median', color=yc, marker=marker)
                ax.fill_between(use_x, np.nanpercentile(usey, 25, axis=1), np.nanpercentile(usey, 75, axis=1),
                                color=yc, alpha=0.25)
                # just to provide correct label
                ax.fill_between(use_x[0:1], np.nanpercentile(usey, 25, axis=1)[0:1],
                                np.nanpercentile(usey, 25, axis=1)[0:1],
                                color=yc, alpha=0.50, label='25-75%')
                ax.fill_between(use_x, np.nanpercentile(usey, 5, axis=1), np.nanpercentile(usey, 95, axis=1),
                                label='5-95%', color=yc, alpha=0.25)

            ax.set_ylabel(self.long_name_dict[y])
        if twin_axs:
            ax = axs[0]
        else:
            ax = axs[-1]
        ax.set_xlabel(xlabel)
        if twin_axs:
            handles, labels = axs[0].get_legend_handles_labels()
            for ax in axs[1:]:
                h, l = ax.get_legend_handles_labels()
                handles += h
                labels += l
            ax.legend(handles, labels)
        else:
            ax.legend()
        return fig, axs

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_production(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_next_state(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_sup_feed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_running_cost(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def reset_state(self, i_month, ):
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')


class DummySimpleFarm(BaseSimpleFarmModel):
    states = {  # i value: (nmiking, stock levels)
        1: ('2aday', 'low'),
        2: ('2aday', 'norm'),
        3: ('1aday', 'low'),
        4: ('1aday', 'norm'),
    }

    month_reset = 7  # trigger farm reset on day 1 in July

    def calculate_feed_needed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_production(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_next_state(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_sup_feed(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def calculate_running_cost(self, i_month, month, current_state):
        assert pd.api.types.is_integer(month), f'month must be int, got {type(month)}'
        assert isinstance(current_state, np.ndarray), f'current_state must be np.ndarray, got {type(current_state)}'
        assert current_state.shape == (
            self.model_shape[1],), f'current_state must be shape {self.model_shape[1]}, got {current_state.shape}'
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')

    def reset_state(self, i_month, ):
        out = np.zeros(self.model_shape[1])
        assert out.shape == (self.model_shape[1],), f'out must be shape {self.model_shape[1]}, got {out.shape}'
        raise NotImplementedError('must be set in a child class')


# todo make child class for dairy, dairy support, beef & sheep

def get_colors(vals, cmap_name='tab20'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = get_cmap(cmap_name)
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = get_cmap(cmap_name)
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


month_len = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}
