"""
created matt_dumont 
on: 10/6/24
"""
import numpy as np
import matplotlib.pyplot as plt
from komanawa.simple_farm_model import SimpleDairyModel
from komanawa.simple_farm_model.simple_dairy_model import mj_per_kg_dm
from op_expenses import get_operating_expenses


def _plot_outputs(outdata,
                  stocking_rates, use_sup, ms_prod, expect_feed_demand,
                  xs, gross_rev, net_income, opt_expese,
                  study_name,
                  rel_lab,
                  base_x,
                  xticklabs,

                  plot_rel=False, plot_money_rel=True,
                  ):
    rel_sr = stocking_rates[base_x]
    nanv = np.full_like(stocking_rates, np.nan)
    keys = 'total_prod', 'last_feed', 'feed_import', 'last_lact', 'total_feed_demand', 'prod_per_cow'
    nice_names = 'Total production', 'Feed Stored', 'Feed Imported', 'Fraction lactating Cows (EOY)', 'Total Feed Demand', 'Production per cow'
    units = 'kg MS', 'kg DM', 'kg DM', 'fraction', 'kg DM', 'kg MS'
    reported = ms_prod, nanv, use_sup, nanv, expect_feed_demand, ms_prod / stocking_rates
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.05, 1, 1])
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis('off')
    axs = []
    xplot = np.arange(len(stocking_rates))
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r + 1, c])
            axs.append(ax)
    for key, rport, ax, nm, u in zip(keys, reported, axs, nice_names, units):
        got = outdata.loc[xs, key] / outdata.loc[base_x, key]
        expect = rport[xs] / rport[base_x]

        if plot_rel:
            ax_raw = ax.twinx()
            if key != 'last_dry':
                ax_raw.plot(xplot, got, label=f'Modelled relative to {rel_lab} SR: {rel_sr} ' + '$cows~ha^{-1}$', c='r',
                            alpha=0.5, ls='--', )
            ax_raw.plot(xplot, expect, label=f'{study_name} relative to {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$',
                        c='b', ls='--',
                        alpha=0.5)
            ax_raw.set_ylabel(f'Relative to {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$')
        ax.set_ylabel(f'Relative to {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$')
        ax.set_xticks(xplot)
        ax.set_xticklabels(xticklabs, rotation=-20)

        got = outdata.loc[xs, key]
        expect = rport[xs]

        ax.set_ylabel(f'Absolute ({u})')
        ax.plot(xplot, got, label='Modelled', c='r', alpha=0.5, marker='o')
        ax.plot(xplot, expect, label=study_name, c='b', alpha=0.5, marker='o')
        if key == 'total_feed_demand':
            got = outdata.loc[xs, 'lact_feed_demand']
            ax.plot(xplot, got, label='Modelled Lact. Feed Demand', c='r', alpha=0.5, marker='.', ls=':')
        ax.set_title(nm)

    ax = axs[-2]
    hand, labs = ax.get_legend_handles_labels()
    if plot_rel:
        hand_raw, labs_raw = ax_raw.get_legend_handles_labels()
    else:
        hand_raw, labs_raw = [], []
    leg_ax.legend(hand + hand_raw, labs + labs_raw, loc='center', ncol=2)
    fig.supxlabel('Stocking rate (cows/ha)')

    fig.tight_layout()

    keys = 'total_prod_money', 'net_income', 'total_expenses', 'total_feed_cost', 'feed_handling_cost', 'op_expenses'
    nice_names = 'Total production value', 'Net income', 'Total expenses', 'Total feed cost', 'Feed handling cost', 'Operating expenses'
    units = '$', '$', '$', '$', '$', '$'
    reported = gross_rev, net_income, opt_expese, nanv, nanv, opt_expese

    fig_money = plt.figure(figsize=(14, 10))
    gs = fig_money.add_gridspec(3, 3, height_ratios=[0.1, 1, 1])
    leg_ax = fig_money.add_subplot(gs[0, :])
    leg_ax.axis('off')
    moneyaxs = []
    xplot = np.arange(len(stocking_rates))
    for r in range(2):
        for c in range(3):
            ax = fig_money.add_subplot(gs[r + 1, c])
            moneyaxs.append(ax)
    for key, rport, ax, nm, u in zip(keys, reported, moneyaxs, nice_names, units):
        got = outdata.loc[xs, key] / outdata.loc[base_x, key]
        expect = rport[xs] / rport[base_x]

        if plot_money_rel:
            ax_raw = ax.twinx()
            if key != 'last_dry':
                ax_raw.plot(xplot, got, label=f'Modelled relative to {rel_lab} SR: {rel_sr}' + '$cows~ha^{-1}$', c='r',
                            alpha=0.5, ls='--', )
            ax_raw.plot(xplot, expect, label=f'{study_name} relative tto {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$',
                        c='b', ls='--',
                        alpha=0.5)
            ax_raw.set_ylabel(f'Relative to {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$')
        ax.set_ylabel(f'Relative to {rel_lab} SR: {rel_sr}' + ' $cows~ha^{-1}$')
        ax.set_xticks(xplot)
        ax.set_xticklabels(xticklabs, rotation=-30)

        got = outdata.loc[xs, key]
        expect = rport[xs]
        ax.set_ylabel(f'Absolute ({u})')
        ax.plot(xplot, got, label='Modelled', c='r', alpha=0.5, marker='o')
        ax.plot(xplot, expect, label=study_name, c='b', alpha=0.5, marker='o')
        ax.set_title(nm)

    hand, labs = ax.get_legend_handles_labels()
    if plot_money_rel:
        hand_raw, labs_raw = ax_raw.get_legend_handles_labels()
    else:
        hand_raw, labs_raw = [], []
    leg_ax.legend(hand + hand_raw, labs + labs_raw, loc='center', ncol=2)
    fig_money.supxlabel('Stocking rate (cows/ha)')

    fig_money.tight_layout()

    return fig, axs, fig_money, moneyaxs


def _extract_data(dm, outdata, i, stock_rate, land_modifer=1, cow_modifer=1, add_sup_cost=0,
                  additional_stock_modifier=1, expense_modifier=1):
    assert issubclass(type(dm), SimpleDairyModel)
    tprod = np.nansum(dm.model_prod) / land_modifer
    last_lact = dm.out_lactating_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
    last_dry = dm.out_dry_cow_fraction[np.where((dm.all_days == 31) & (dm.all_months == 5))][0, 0]
    last_feed = dm.model_feed[-1][0] / mj_per_kg_dm
    feed_import = dm.model_feed_imported.sum() / mj_per_kg_dm
    total_feed_cost = dm.model_feed_cost.sum() + add_sup_cost
    outdata.loc[i, 'total_prod'] = tprod
    outdata.loc[i, 'prod_per_cow'] = tprod / stock_rate
    outdata.loc[i, 'last_lact'] = last_lact
    outdata.loc[i, 'last_dry'] = last_dry
    outdata.loc[i, 'last_feed'] = last_feed
    outdata.loc[i, 'feed_import'] = feed_import
    outdata.loc[i, 'lact_feed_demand'] = np.nansum(dm.out_feed_lactating) / mj_per_kg_dm / land_modifer
    outdata.loc[i, 'total_feed_demand'] = np.nansum(dm.model_feed_demand) / mj_per_kg_dm
    outdata.loc[i, 'total_prod_money'] = np.nansum(dm.model_prod_money) / land_modifer
    outdata.loc[i, 'money'] = dm.model_money[-1] / land_modifer
    outdata.loc[i, 'total_feed_cost'] = total_feed_cost
    outdata.loc[i, 'feed_handling_cost'] = (np.nansum(dm.model_prod_money) - dm.model_money[-1] -
                                            dm.model_feed_cost.sum())
    outdata.loc[i, 'op_expenses'] = sum(list(get_operating_expenses(
        stock_rate,
        addition_stock_modifer=additional_stock_modifier).values())) * expense_modifier
    outdata.loc[i, 'total_expenses'] = outdata.loc[i, 'total_feed_cost'] + outdata.loc[i, 'op_expenses'] + \
                                       outdata.loc[i, 'feed_handling_cost']
    outdata.loc[i, 'net_income'] = outdata.loc[i, 'money'] - outdata.loc[i, 'op_expenses']
