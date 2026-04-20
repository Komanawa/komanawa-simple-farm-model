"""
created matt_dumont 
on: 10/13/24
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from ludf import raw_full_farm_run as ludf_ff
from ludf import dairy_platform_run as ludf_mp
from kok_et_al_2022 import unmodified_comparison as kok_ff
from kok_et_al_2022 import make_data_silage_ifeed as kok_mp
from macdonal_2011 import run_farm_model_no_mod as mac_ff
from macdonal_2011 import run_farm_model as mac_mp
from pathlib import Path


# total production
# net income
# total expenses
# total production
# feed imported??
# feed demand
# color = Farms:  M. 2.2, M. 2.7, M. 3.1, M. 3.7, M. 4.3,
# rosybrown, indianred, firebrick, red, maroon
# K. LSR, K. LUDF, K. MSR,
# cornflowerblue, blue, navy
#
# LUDA
# darkorange
# symbol = full system vs milking platform

def make_summary_plot():
    h = 8
    hr = [0.2, 1, 1]

    fig = plt.figure(figsize=(3 / sum(hr) * h, h))
    gs = fig.add_gridspec(3, 3, height_ratios=hr)
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis('off')
    axs = []
    labels = ['M. 2.2', 'M. 2.7', 'M. 3.1', 'M. 3.7', 'M. 4.3', 'K. LSR', 'K. LUDF', 'K. MSR', 'LUDF-MY']
    colors = ['rosybrown', 'indianred', 'firebrick', 'red', 'maroon', 'cornflowerblue', 'blue', 'navy', 'darkorange']
    mlabs = ['Full system', 'Milking platform']
    markers = ["^", "o"]
    handles = []
    for ml, m in zip(mlabs, markers):
        handles.append(plt.Line2D([0], [0], marker=m, color='w', label=ml, markersize=10, markerfacecolor='k'))
    for l, c in zip(labels, colors):
        handles.append(plt.Line2D([0], [0], color='w', label=l, markersize=10, markerfacecolor=c, marker='s'))
    handles.append(plt.Line2D([0, 1], [0, 1], c='k', ls=':', alpha=0.25, label='1:1 line'))
    leg_ax.legend(handles=handles, loc='center', ncol=6)
    for i in range(2):
        for j in range(3):
            axs.append(fig.add_subplot(gs[i + 1, j]))

    titles = ['Production\n($kgMS~ha^{-1}$)',
              'Feed demand\n($kgDM~ha^{-1}$)',
              'Feed imported\n($kgDM~ha^{-1}$)',
              'Gross ($ \\$~ha^{-1}$)',
              'Expenses ($ \\$~ha^{-1}$)',
              'Net ($ \\$~ha^{-1}$)',
              ]
    keys = [
        'prod',
        'feed_d',
        'feed_i',
        'gross',
        'exp',
        'net',
    ]

    lims = {k: [] for k in keys}

    for marker, lf, mf, kf in zip(markers, [ludf_ff, ludf_mp], [mac_ff, mac_mp], [kok_ff, kok_mp]):

        ludf = lf(False)
        ludf_labs = [2021, 2022, 2023]
        ludf_colors = ['darkorange'] * 3

        kok = kf()
        kok_labs = ['LSR-1', 'LUDF-1', 'MSR-1', 'LSR-2', 'LUDF-2', 'MSR-2', ]
        kok_colors = ['cornflowerblue', 'blue', 'navy', 'cornflowerblue', 'blue', 'navy']

        mac = mf()
        mac_labs = ['M. 2.2', 'M. 2.7', 'M. 3.1', 'M. 3.7', 'M. 4.3']
        mac_colors = ['rosybrown', 'indianred', 'firebrick', 'red', 'maroon']
        for labs, df, colors in zip([ludf_labs, mac_labs, kok_labs], [ludf, mac, kok],
                                    [ludf_colors, mac_colors, kok_colors]):
            for lab, color in zip(labs, colors):
                for ax, key in zip(axs, keys):
                    x = df.loc[lab, f'{key}_mod']
                    y = df.loc[lab, f'{key}_mes']
                    lims[key].extend([x, y])
                    ax.scatter(x, y, color=color, marker=marker)

    # make r2 values and add to titles
    use_titles = []
    for key, basetitle in zip(keys, titles):
        meas = []
        mod = []
        study = []
        for df, st in zip([ludf, mac, kok], ['ludf', 'mac', 'kok']):
            meas.extend(df[f'{key}_mes'].values)
            mod.extend(df[f'{key}_mod'].values)
            study.extend([st]*len(df))
        meas = np.array(meas)
        mod = np.array(mod)
        study = np.array(study)
        r2 = 1 - (np.nansum((meas - mod) ** 2) / np.nansum((meas - np.nanmean(meas)) ** 2))
        if key in ['exp','net']:
            meas_m = meas[study != 'mac']
            mod_m = mod[study != 'mac']
            r2_wo_mac = 1 - (np.nansum((meas_m - mod_m) ** 2) / np.nansum((meas_m - np.nanmean(meas_m)) ** 2))
            use_titles.append(f'{basetitle}\n'+'$R^2_{1:1}$'+f'={r2:.2f}\n'+'$R^2_{1:1, EM}$'+f'={r2_wo_mac:.2f}')
        else:
            use_titles.append(f'{basetitle}\n'+'$R^2_{1:1}$'+f'={r2:.2f}')

    for t, ax, l in zip(use_titles, axs, keys):
        ll = np.nanmin(lims[l]) - np.nanmax(lims[l]) * 0.05
        ul = np.nanmax(lims[l]) * 1.05

        ax.set_xlim(ll, ul)
        ax.set_ylim(ll, ul)
        ax.set_aspect('equal')

        ax.plot([0, 1], [0, 1], c='k', ls=':', alpha=0.25, transform=ax.transAxes, zorder=-1)
        ax.text(0.05, 0.95, t, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5), va='top', ha='left')

    fig.supxlabel('Modelled (This work)')
    fig.supylabel('Reported (Observed - FARMAX model)')
    fig.tight_layout()
    plt.show()
    fig.savefig(Path(__file__).parent.joinpath('summary_plot.png'), dpi=600)


def print_rmses():
    titles = ['Production\n($kgMS~ha^{-1}$)',
              'Feed demand\n($kgDM~ha^{-1}$)',
              'Feed imported\n($kgDM~ha^{-1}$)',
              'Gross\n($ \\$~ha^{-1}$)',
              'Expenses\n($ \\$~ha^{-1}$)',
              'Net\n($ \\$~ha^{-1}$)',
              ]
    keys = [
        'prod',
        'feed_d',
        'feed_i',
        'gross',
        'exp',
        'net',
    ]

    rmses = pd.DataFrame(dtype=float)
    mae = pd.DataFrame(dtype=float)
    mean_observed = pd.DataFrame(dtype=float)
    for lf, mf, kf in zip([ludf_ff, ludf_mp], [mac_ff, mac_mp], [kok_ff, kok_mp]):
        ludf = lf(False)
        kok = kf()
        mac = mf()
        all_data = pd.concat([ludf, mac, kok], axis=0)
        all_wo_mac = pd.concat([ludf, kok], axis=0)

        for key, nicekey in zip(keys, titles):
            nicekey = nicekey.split('\n')[0]  # remove units for the column name
            for df, name in zip([ludf, mac, kok, all_data, all_wo_mac],
                                ['LUDF', 'Mac', 'Kok', 'All', 'All ex. Mac']):

                x = df[f'{key}_mod']
                y = df[f'{key}_mes']
                rmse = np.sqrt(np.nanmean((x - y) ** 2))
                rmses.loc[name, nicekey] = rmse
                mae_val = np.nanmean(np.abs(x - y))
                mae.loc[name, nicekey] = mae_val
                mean_observed.loc[name, nicekey] = np.nanmean(y)
    rmses = rmses.round(2)
    print('RMSEs:')
    print(rmses)
    print('MAEs:')
    print(mae)
    from komanawa.ksl_tools.writeup_support import make_latex_table

    s = make_latex_table(None, rmses, frac_txt_width=0.85, table_loc=None, label='rmse',
                     caption='Root mean squared errors of the farm model validation',
                         float_format='{:,.2f}', ksltable=False, )
    with Path.home().joinpath('Downloads','rmse_table.tex').open('w') as f:
        f.write('% made from komanawa-simple-farm-model.validation_overview.print_rmses\n')
        f.write(s)
    s = make_latex_table(None, mae, frac_txt_width=0.85, table_loc=None, label='mae',
                     caption='Mean absolute errors of the farm model validation',
                         float_format='{:,.2f}', ksltable=False, )
    with Path.home().joinpath('Downloads','mae_table.tex').open('w') as f:
        f.write('% made from komanawa-simple-farm-model.validation_overview.print_rmses\n')
        f.write(s)

    s = make_latex_table(None, mean_observed, frac_txt_width=0.85, table_loc=None, label='mean_observed',
                     caption='Mean observed values of the farm model validation',
                         float_format='{:,.2f}', ksltable=False, )
    with Path.home().joinpath('Downloads','mean_observed_table.tex').open('w') as f:
        f.write('% made from komanawa-simple-farm-model.validation_overview.print_rmses\n')
        f.write(s)


if __name__ == '__main__':
    #make_summary_plot()
    print_rmses()
