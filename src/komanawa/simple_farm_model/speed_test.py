"""
created matt_dumont 
on: 8/30/24
"""
from komanawa.simple_farm_model import DairyModelWithSCScarcity
import pandas as pd
import time
from pathlib import Path
from komanawa.simple_farm_model import default_peak_cow
import numpy as np


# todo run this...

def speed_test(savedir=None):
    steps = [None, 0.05, 0.1, 0.15, 0.2, 5, 10, 15, 20, 30, 50]
    # note that for coarse mode, the step is used as the cull/dryoff level arg
    opt_modes = ['optimised', 'step', 'step', 'step', 'step', ] + ['coarse'] * 6
    step_labs = [f'{o}-{e}' for e, o in zip(steps, opt_modes)]
    percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    outdata_money = dict()
    outdata_imported = dict()

    outdata_speed = pd.DataFrame(index=step_labs)
    for lab, step, opt_mode in zip(step_labs, steps, opt_modes):
        t = time.time()
        (model_name, percent_imported, model_money, feed_cost, scar_cost, lac_frac,
         dry_frac, model_feed) = run_model(step, opt_mode)
        print(f'{model_name=}, {step=} took {time.time() - t} seconds', )
        outdata_money[(lab, model_name)] = model_money
        outdata_imported[(lab, model_name)] = percent_imported
        outdata_speed.loc[lab, model_name] = time.time() - t
    outdata_money = {k: outdata_money[k] / outdata_money[('optimised-None', 'DairyModelWithSCScarcity')] for k in
                     outdata_money}
    outdata_imported = {k: outdata_imported[k] / outdata_imported[('optimised-None', 'DairyModelWithSCScarcity')] for k
                        in
                        outdata_imported}
    outdata_money = pd.DataFrame(outdata_money).describe(percentiles)
    outdata_imported = pd.DataFrame(outdata_imported).describe(percentiles)
    if savedir is not None:
        savedir = Path(savedir)
        savedir.mkdir(exist_ok=True)
        outdata_money.round(3).to_csv(savedir.joinpath('speed_test_money.csv'))
        outdata_imported.round(3).to_csv(savedir.joinpath('speed_test_imported.csv'))
        outdata_speed.round(2).to_csv(savedir.joinpath('speed_test_speed.csv'))
    print(outdata_speed)
    print(outdata_money)
    print(outdata_imported)


def run_model(instep, opt_mode):
    mj_per_kg_dm = 11  # MJ ME /kg DM
    sup_cost = 406 / 1000 / mj_per_kg_dm
    product_price = 8.09
    all_months = np.array([(7 - 1 + i) % 12 + 1 for i in range(12)])
    s, a, b, c = (11.4713, 2.0711, 1.9700, 20.4965)

    inpg = np.load(Path(__file__).parent.joinpath('test_data', 'eyrewell_pg.npz'))['inpg']
    nsims = inpg.shape[1]
    if opt_mode == 'coarse':
        cull_level = dryoff_level = instep
        step = None
    else:
        cull_level = dryoff_level = None
        step = instep
    model = DairyModelWithSCScarcity(all_months,
                                     istate=[0] * nsims, pg=inpg, ifeed=[0] * nsims,
                                     imoney=[0] * nsims,
                                     sup_feed_cost=sup_cost,
                                     product_price=product_price,
                                     monthly_input=True, ncore_opt=None,
                                     cull_dry_step=step, opt_mode=opt_mode,
                                     cull_levels=cull_level, dryoff_levels=dryoff_level,
                                     s=s, a=a, b=b, c=c, )
    model.run_model(printi=False)
    needed_feed = model.get_annual_feed()
    imported_feed = model.cum_feed_import[-1]
    percent_imported = imported_feed / needed_feed * 100
    idx = ~np.isin(model.all_months, [5, 6, 7])

    return (DairyModelWithSCScarcity.__name__,
            percent_imported,
            model.model_money[-1],
            np.nansum(model.model_feed_cost, axis=0),
            np.nansum(model.feed_scarcity_cost, axis=0),
            np.nanmean(model.out_lactating_cow_fraction[idx, :], axis=0),
            np.nanmean(model.out_dry_cow_fraction[idx, :], axis=0),
            model.model_feed[-1],
            )


if __name__ == '__main__':
    speed_test(Path.home().joinpath('Downloads', 'farm_speed_test'))
