"""
created matt_dumont 
on: 10/13/24
"""
import numpy as np
import matplotlib.pyplot as plt
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

    for t, ax, l in zip(titles, axs, keys):
        ll = np.nanmin(lims[l]) - np.nanmax(lims[l]) * 0.05
        ul = np.nanmax(lims[l]) * 1.05

        ax.set_xlim(ll, ul)
        ax.set_ylim(ll, ul)
        ax.set_aspect('equal')

        ax.plot([0, 1], [0, 1], c='k', ls=':', alpha=0.25, transform=ax.transAxes, zorder=-1)
        ax.text(0.05, 0.95, t, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5), va='top', ha='left')

    fig.supxlabel('Modelled (This work)')
    fig.supylabel('Reported (Observed - Farmmax model)')
    fig.tight_layout()
    fig.savefig(Path(__file__).parent.joinpath('summary_plot.png'), dpi=600)
    plt.show()


if __name__ == '__main__':
    make_summary_plot()
