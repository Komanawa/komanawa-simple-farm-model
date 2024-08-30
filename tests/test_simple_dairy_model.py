"""
created matt_dumont 
on: 6/26/24
"""
import unittest
from pathlib import Path
import numpy as np
from komanawa.simple_farm_model.simple_dairy_model import SimpleDairyModel, DairyModelWithSCScarcity, default_peak_cow

testdatadir = Path(__file__).parent.joinpath('test_data')


# keynote the saved tests were only loosely checked, they may not be correct

class TestSimpleDairyModel(unittest.TestCase):
    def get_eyrewell_test_data(self):
        """
        get the eyrewell test data (for the simple dairy model) and/or the DairyModelWithSCScarcity
        :return:
        """
        mj_per_kg_dm = 11  # MJ ME /kg DM
        sup_cost = 406 / 1000 / mj_per_kg_dm
        product_price = 8.09
        all_months = np.array([(7 - 1 + i) % 12 + 1 for i in range(12)])
        inpg = np.load(Path(__file__).parent.joinpath('test_data', 'eyrewell_pg.npz'))['inpg']
        nsims = inpg.shape[1]
        out = dict(s=np.int32(10), a=1, b=np.float32(0.9), c=16, sup_cost=sup_cost, product_price=product_price,
                   all_months=all_months,
                   nsims=nsims, pg=inpg, peak_cow=np.float32(default_peak_cow))
        return out

    def test_init_no_pg(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        model = SimpleDairyModel(all_months,
                                 istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                                 sup_feed_cost=1, product_price=10, monthly_input=True)
        model.run_model(printi=False)

    def test_model_runs(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        model = SimpleDairyModel(all_months,
                                 istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                                 sup_feed_cost=1, product_price=10, monthly_input=True)
        model.run_model(printi=False)

    def test_model_mp_no_mp(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        no_mp = SimpleDairyModel(all_months,
                                 istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                                 sup_feed_cost=1, product_price=10, monthly_input=True)
        no_mp.run_model(printi=False)
        mp = SimpleDairyModel(all_months,
                              istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                              sup_feed_cost=1, product_price=10, monthly_input=True, ncore_opt=2)
        mp.run_model(printi=False)
        no_mp.output_eq(mp, raise_on_diff=True, skip_alt_kwargs=['ncore_opt'])

    def test_model_stepsize(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        model = SimpleDairyModel(all_months,
                                 istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                                 sup_feed_cost=1, product_price=10, opt_mode='step',
                                 monthly_input=True, cull_dry_step=0.01)
        model.run_model(printi=False)

    def test_model_coarse_opt(self):
        all_months = [(7 - 1 + i) % 12 + 1 for i in range(12)] * 3
        model = SimpleDairyModel(all_months,
                                 istate=[0, 0], pg=[0] * len(all_months), ifeed=[0, 0], imoney=[0, 0],
                                 sup_feed_cost=1, product_price=10, opt_mode='coarse',
                                 monthly_input=True, cull_dry_step=None,
                                 cull_levels=10, dryoff_levels=10)
        model.run_model(printi=False)

    def test_model_with_data(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        model = SimpleDairyModel(input_data['all_months'],
                                 istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims, imoney=[0] * nsims,
                                 sup_feed_cost=input_data['sup_cost'],
                                 product_price=input_data['product_price'],
                                 monthly_input=True, ncore_opt=None, peak_lact_cow_per_ha=input_data['peak_cow'])
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = SimpleDairyModel.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_with_data_SC(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data_SC.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        s = input_data['s']
        a = input_data['a']
        b = input_data['b']
        c = input_data['c']
        model = DairyModelWithSCScarcity(input_data['all_months'],
                                         istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                         imoney=[0] * nsims,
                                         sup_feed_cost=input_data['sup_cost'],
                                         product_price=input_data['product_price'],
                                         monthly_input=True, ncore_opt=None,
                                         s=s, a=a, b=b, c=c, )
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = DairyModelWithSCScarcity.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_stepsize_data(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data_step.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        model = SimpleDairyModel(input_data['all_months'],
                                 istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims, imoney=[0] * nsims,
                                 sup_feed_cost=input_data['sup_cost'],
                                 product_price=input_data['product_price'],
                                 opt_mode='step',
                                 monthly_input=True, ncore_opt=None, peak_lact_cow_per_ha=input_data['peak_cow'],
                                 cull_dry_step=0.1)
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = SimpleDairyModel.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_coarse_data(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data_coarse.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        model = SimpleDairyModel(input_data['all_months'],
                                 istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims, imoney=[0] * nsims,
                                 sup_feed_cost=input_data['sup_cost'],
                                 product_price=input_data['product_price'],
                                 opt_mode='coarse',
                                 monthly_input=True, ncore_opt=None, peak_lact_cow_per_ha=input_data['peak_cow'],
                                 cull_dry_step=None,
                                 cull_levels=10,
                                 dryoff_levels=10)
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = SimpleDairyModel.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_stepsize_data_SC(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data_SC_step.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        s = input_data['s']
        a = input_data['a']
        b = input_data['b']
        c = input_data['c']
        model = DairyModelWithSCScarcity(input_data['all_months'],
                                         istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                         imoney=[0] * nsims,
                                         sup_feed_cost=input_data['sup_cost'],
                                         product_price=input_data['product_price'],
                                         opt_mode='step',
                                         monthly_input=True, ncore_opt=None,
                                         s=s, a=a, b=b, c=c, cull_dry_step=0.1)
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = DairyModelWithSCScarcity.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_coarse_opt_data_SC(self):
        save_data = False
        save_path = testdatadir.joinpath('expect_test_model_with_data_SC_coarse_opt.nc')
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        s = input_data['s']
        a = input_data['a']
        b = input_data['b']
        c = input_data['c']
        model = DairyModelWithSCScarcity(input_data['all_months'],
                                         istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                         imoney=[0] * nsims,
                                         sup_feed_cost=input_data['sup_cost'],
                                         product_price=input_data['product_price'],
                                         opt_mode='coarse',
                                         monthly_input=True, ncore_opt=None,
                                         s=s, a=a, b=b, c=c, cull_dry_step=None,
                                         cull_levels=10, dryoff_levels=10)
        model.run_model(printi=False)
        if save_data:
            model.save_results_nc(save_path)
        expect = DairyModelWithSCScarcity.from_input_file(save_path)
        expect.output_eq(model, raise_on_diff=True)

    def test_model_mp_no_mp_data(self):
        input_data = self.get_eyrewell_test_data()
        input_data_mp = input_data.copy()
        nsims = input_data['nsims']
        model = SimpleDairyModel(input_data['all_months'],
                                 istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims, imoney=[0] * nsims,
                                 sup_feed_cost=input_data['sup_cost'],
                                 product_price=input_data['product_price'],
                                 monthly_input=True, ncore_opt=1, peak_lact_cow_per_ha=input_data['peak_cow'])
        model.run_model(printi=False)
        model_mp = SimpleDairyModel(input_data_mp['all_months'],
                                    istate=[0] * nsims, pg=input_data_mp['pg'], ifeed=[0] * nsims, imoney=[0] * nsims,
                                    sup_feed_cost=input_data_mp['sup_cost'],
                                    product_price=input_data_mp['product_price'],
                                    monthly_input=True, ncore_opt=1, peak_lact_cow_per_ha=input_data_mp['peak_cow'])
        model_mp.run_model(printi=False)
        model.output_eq(model_mp, raise_on_diff=True)

    def test_model_mp_no_mp_data_SC(self):
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        s = input_data['s']
        a = input_data['a']
        b = input_data['b']
        c = input_data['c']
        model = DairyModelWithSCScarcity(input_data['all_months'],
                                         istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                         imoney=[0] * nsims,
                                         sup_feed_cost=input_data['sup_cost'],
                                         product_price=input_data['product_price'],
                                         monthly_input=True, ncore_opt=1,
                                         s=s, a=a, b=b, c=c, )
        model.run_model(printi=False)

        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        s = input_data['s']
        a = input_data['a']
        b = input_data['b']
        c = input_data['c']
        model_mp = DairyModelWithSCScarcity(input_data['all_months'],
                                            istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                            imoney=[0] * nsims,
                                            sup_feed_cost=input_data['sup_cost'],
                                            product_price=input_data['product_price'],
                                            monthly_input=True, ncore_opt=None,
                                            s=s, a=a, b=b, c=c, )
        model_mp.run_model(printi=False)
        model.output_eq(model_mp, raise_on_diff=True, skip_alt_kwargs=['ncore_opt'])

    def test_weird_failure(self):
        input_data = self.get_eyrewell_test_data()
        nsims = input_data['nsims']
        peak_cow, s, a, b, c = [18.03014198, 48.21515981, 2.09236717, 2.16710216, 10.12209119]
        model = DairyModelWithSCScarcity(input_data['all_months'],
                                         istate=[0] * nsims, pg=input_data['pg'], ifeed=[0] * nsims,
                                         imoney=[0] * nsims,
                                         peak_lact_cow_per_ha=peak_cow,
                                         sup_feed_cost=input_data['sup_cost'],
                                         product_price=input_data['product_price'],
                                         monthly_input=True, ncore_opt=1,
                                         s=s, a=a, b=b, c=c, )
        model.run_model(printi=False)
