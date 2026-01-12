"""
This script does the plotting.
"""
from utils import get_desktop_folder, save_safely
import os
import sciris as sc
import pandas as pd
import seaborn as sns
from copy import deepcopy as dcp
from matplotlib import pyplot as plt
from constants import *

desktop_folder = get_desktop_folder()

# Plotting parameters
SENS_PLOT_KIND = 'bar'  # Sensitivity plot kind (usually one of 'bar' or 'box')
ERRWIDTH = 1.0  # Error bar line width
CAPSIZE = 0.5  # Error bar cap width
DIAG_PLOTS = False
plt.rcParams['figure.dpi'] = 300


def prepare_df_for_plotting(results, which_df=None):
    if which_df is None: which_df = results.df
    df = dcp(which_df)

    keep_cols = ['scen_name', 'dag_name', 'nodekey', 'outcome', 'year', 'estimate', 'value', 'cost', 'cost_cat', 'outcome_cat', 'node_long', 'scen_long']
    df = df[[col for col in keep_cols if col in df.columns]]
    df = df.loc[df['outcome'] != '!TOTAL-COST']
    df = df.loc[df['outcome'] != '!TOTAL-OUTCOME']

    df['node_long'] = df.apply(lambda row: results.node_info[(row.dag_name, row.nodekey)]['long'], axis=1)
    df['scen_long'] = df.apply(lambda row: results.scenarios[row.scen_name].long, axis=1)

    df = custom_df_changes(df)

    # Sum outputs over the years, by dropping the year so they match to the same, then sum
    df = df.drop(['year'], axis=1)

    value_columns = ['value', 'cost']
    df = df.groupby([col for col in df.columns if not col in value_columns], sort=False, dropna=False)[value_columns].sum().reset_index()

    return df

def custom_df_changes(data):
    df = dcp(data)
    df['scen_long'] = df['scen_long'].str.replace(pat='peerled', repl='nurse/peer-led')
    return df

def format_y_axes_label_costs(g):
    # format y-axis labels
    for ax in g.axes.flat:
        format_y_axes_label_costs_single(ax)

def format_y_axes_label_costs_single(ax):
    max_val = np.max(ax.get_ylim())
    if max_val >= 8e6:
        ax.yaxis.set_major_formatter(lambda y, p: f'AU${y / 1e6:1.1f}M')
    elif max_val >= 0.8e6:
        ax.yaxis.set_major_formatter(lambda y, p: f'AU${y / 1e6:1.1f}M')
    elif max_val >= 1e3:
        ax.yaxis.set_major_formatter(lambda y, p: f'AU${y / 1e3:1.1f}K')

def plot_outcomes(data, filename, save_folder=desktop_folder, best_estimate=True, same_axes=True, is_diff=False, sens_plot_kind=SENS_PLOT_KIND):

    df = dcp(data)

    if is_diff:
        title_label = 'Difference from baseline (2026-2030 inclusive)'
    else:
        title_label = 'Estimates of total (2026-2030 inclusive)'

    scen_long_categories = df['scen_long'].unique()

    # Plot relevant outcomes
    df = df.loc[(df['outcome'] == 'yes') | ((df['nodekey'] == 'bbv_treat') & (df['outcome'] == 'treat'))].drop('outcome', axis=1)

    # Filter which nodes to plot
    plot_nodes = PLOT_NODES
    df = df.loc[df['nodekey'].isin(plot_nodes)].reset_index(drop=True)
    df['nodekey'] = pd.Categorical(df['nodekey'], categories=plot_nodes, ordered=True)
    df = df.sort_values(['nodekey']).reset_index(drop=True)

    if is_diff:
        df['value'] = df.groupby(['scen_name', 'outcome_cat'])['value'].transform(
            lambda g: g if (g != 0).any() else np.nan
        )

    df['node_long'] = pd.Categorical(df['node_long'], ordered=True, categories=df['node_long'].unique())
    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=scen_long_categories)

    # split data
    if best_estimate:
        plot_kind = 'bar'
    else:
        plot_kind = sens_plot_kind
    df_best = df.loc[df['estimate'] == 'best'].reset_index(drop=True)
    df_error = df.loc[df['estimate'] != 'best'].reset_index(drop=True)

    # Use Seaborn to plot
    # plt.figure()
    sns.set_theme()
    g = sns.catplot(
        data=df_error, x='scen_long', y='value',
        hue='scen_long',
        col='node_long',
        col_order=list(df['node_long'].cat.categories),
        kind=plot_kind,
        errorbar=('pi', 95),
        err_kws={'linewidth': ERRWIDTH},
        capsize=CAPSIZE,
        col_wrap=6,
        sharey=same_axes,
        legend=True,
        alpha=0.5 if DIAG_PLOTS else 0,
    )
    for i, node_long in enumerate(list(df['node_long'].cat.categories)):
        if not DIAG_PLOTS:
            sns.barplot(
                data=df_best[df_best['node_long'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='value',
                hue='scen_long',
                alpha=1,
                legend=False,
            )
        else:
            sns.swarmplot(
                data=df_best[df_best['node_long'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='value',
                hue='scen_long',
                alpha=1,
                legend=False,
            )

    g.set_titles('{col_name}')
    # g.set_axis_labels('Scenario', 'Total number (2026-2030 inclusive)')
    g.set(ylabel='Total number (2026-2030 inclusive)')
    g.set(xlabel=None)
    g.set(xticklabels=[])
    if not is_diff: g.set(ylim=(0, None))

    sns.move_legend(g, bbox_to_anchor=(5/6, 1/6), loc="center", title=None, fontsize=18)

    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.95)  # adjust the Figure in rp
    g.fig.suptitle(title_label)

    filepath = os.path.join(save_folder, filename)
    plt.savefig(filepath)
    print('Saved:', filepath)

def plot_outcomes_aggregated(data, filename, save_folder=desktop_folder, best_estimate=True, sens_plot_kind=SENS_PLOT_KIND, same_axes=True, is_diff=False):

    df = dcp(data)

    if is_diff:
        title_label = 'Difference from baseline (2026-2030 inclusive)'
    else:
        title_label = 'Estimates of total (2026-2030 inclusive)'

    scen_long_categories = df['scen_long'].unique()

    # Plot relevant outcomes
    df = df.loc[(df['outcome'] == 'yes') | ((df['nodekey'] == 'bbv_treat') & (df['outcome'] == 'treat'))].drop('outcome', axis=1)

    if is_diff:
        df['value'] = df.groupby(['scen_name', 'outcome_cat'])['value'].transform(
            lambda g: g if (g != 0).any() else np.nan
        )

    # df['node_long'] = df['outcome_cat']  # aggregate nodes
    df['node_long'] = df['outcome_cat'].map(OUTCOME_CAT_LABELS)  # aggregate nodes
    df = df.loc[(df['node_long'] != 'none') & (df['node_long'].notnull())]
    df = df.drop(['nodekey', 'cost_cat', 'cost', 'dag_name'], axis=1)
    df = df.groupby([col for col in df.columns if col != 'value'], sort=False, dropna=False)['value'].sum().reset_index()

    df['node_long'] = pd.Categorical(df['node_long'], ordered=True, categories=df['node_long'].unique())
    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=scen_long_categories)

    # split data
    if best_estimate:
        plot_kind = 'bar'
    else:
        plot_kind = sens_plot_kind
    df_best = df.loc[df['estimate'] == 'best'].reset_index(drop=True)
    df_error = df.loc[df['estimate'] != 'best'].reset_index(drop=True)

    # Use Seaborn to plot
    # plt.figure()
    sns.set_theme()
    g = sns.catplot(
        data=df_error, x='scen_long', y='value',
        hue='scen_long',
        col='node_long',
        col_order=list(df['node_long'].cat.categories),
        kind=plot_kind,
        errorbar=('pi', 95),
        err_kws={'linewidth': ERRWIDTH},
        capsize=CAPSIZE,
        col_wrap=3,
        sharey=same_axes,
        legend=True,
        alpha=0.5 if DIAG_PLOTS else 0,
        order=list(df['scen_long'].cat.categories),
    )
    for i, node_long in enumerate(list(df['node_long'].cat.categories)):
        if not DIAG_PLOTS:
            sns.barplot(
                data=df_best[df_best['node_long'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='value',
                hue='scen_long',
                alpha=1,
                legend=False,
            )
        else:
            sns.swarmplot(
                data=df_best[df_best['node_long'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='value',
                hue='scen_long',
                alpha=1,
                legend=False,
            )

    g.set_titles('{col_name}')
    # g.set_axis_labels('Scenario', 'Total number (2026-2030 inclusive)')
    g.set(ylabel='Total number (2026-2030 inclusive)')
    g.set(xlabel=None)
    g.set(xticklabels=[])
    if not is_diff: g.set(ylim=(0, None))

    # sns.move_legend(g, bbox_to_anchor=(5/6, 1/4), loc="center", title=None, fontsize=16)
    sns.move_legend(g, bbox_to_anchor=(0.85, 1/2), loc="center", title=None, fontsize=16)

    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    # g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.fig.subplots_adjust(top=0.9, right=0.70, wspace=0.22)  # adjust the Figure in rp
    g.fig.suptitle(title_label)

    filepath = os.path.join(save_folder, filename)
    plt.savefig(filepath)
    print('Saved:', filepath)

def plot_costs_best(data, save_folder=desktop_folder):
    df = dcp(data)

    df = df.drop(['nodekey', 'node_long', 'outcome', 'value'], axis=1)

    value_columns = ['cost']
    df = df.groupby([col for col in df.columns if not col in value_columns], sort=False, dropna=False)[value_columns].sum().reset_index()

    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=df['scen_long'].unique())

    # split data
    df_best = df.loc[df['estimate'] == 'best'].reset_index(drop=True)

    # best estimate plot
    # plt.figure()
    sns.set_theme()
    g = sns.catplot(
        data=df_best, x='scen_long', y='cost',
        hue='scen_long',
        col='dag_name',
        col_order=list(df['dag_name'].unique()),
        col_wrap=4,
        kind='bar',
        sharey=True,
        legend=True,
    )
    g.set_titles('{col_name}')
    # g.set_axis_labels('Scenario', 'Total costs (2026-2030 inclusive) [$AU]')
    g.set(ylabel='Total costs (2026-2030 inclusive) [$AU]')
    g.set(xlabel=None)
    g.set(xticklabels=[])
    g.set(ylim=(0, None))

    sns.move_legend(g, bbox_to_anchor=(5/6, 1/6), loc="center", title=None, fontsize=26)

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.90)  # adjust the Figure in rp
    g.fig.suptitle('Estimates of costs (2026-2030 inclusive)')

    filepath = os.path.join(save_folder, f'fig_costs_{TODAY_STR}.png')
    plt.savefig(filepath)
    print('Saved:', filepath)

def plot_costs_categorised(data, filename, save_folder=desktop_folder, best_estimate=True, sens_plot_kind=SENS_PLOT_KIND, same_axes=True, is_diff=False, **kwargs):

    df = dcp(data)

    if is_diff:
        title_label = 'Difference in costs from baseline (2026-2030 inclusive)'
    else:
        title_label = 'Estimates of costs (2026-2030 inclusive)'

    # Summarise costs
    # df = df.drop(['nodekey', 'node_long', 'dag_name', 'outcome', 'value'], axis=1)
    df = df.groupby(['scen_name', 'estimate', 'cost_cat', 'scen_long'], sort=False)['cost'].sum().reset_index()

    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=df['scen_long'].unique())
    df['cost_cat_label'] = df['cost_cat'].map(COST_CAT_LABELS)

    # Split data
    if best_estimate:
        plot_kind = 'bar'
    else:
        plot_kind = sens_plot_kind
    df_best = df.loc[df['estimate'] == 'best'].reset_index(drop=True)
    df_error = df.loc[df['estimate'] != 'best'].reset_index(drop=True)

    # Use Seaborn to plot
    # plt.figure()
    sns.set_theme()
    g = sns.catplot(
        data=df_error, x='scen_long', y='cost',
        hue='scen_long',
        col='cost_cat_label',
        col_order=COST_CAT_LABELS.values(),
        col_wrap=3,
        kind=plot_kind,
        errorbar=('pi', 95),
        err_kws={'linewidth': ERRWIDTH},
        capsize=CAPSIZE,
        sharey=same_axes,
        legend=True,
        alpha=0.5 if DIAG_PLOTS else 0,
    )
    for i, node_long in enumerate(COST_CAT_LABELS.keys()):
        if not DIAG_PLOTS:
            sns.barplot(
                data=df_best[df_best['cost_cat'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='cost',
                hue='scen_long',
                alpha=1,
                legend=False,
            )
        else:
            sns.swarmplot(
                data=df_best[df_best['cost_cat'] == node_long],  # select the relevant data subset
                ax=g.axes[i],  # append to the relevant axis
                x='scen_long',
                y='cost',
                hue='scen_long',
                alpha=1,
                legend=False,
            )

    g.set_titles('{col_name}')
    # g.set_axis_labels('Scenario', 'Total costs (2026-2030 inclusive)')
    g.set(ylabel='Total costs (2026-2030 inclusive)')
    g.set(xlabel=None)
    g.set(xticklabels=[])
    if not is_diff: g.set(ylim=(0, None))

    # format y-axis labels
    format_y_axes_label_costs(g)

    sns.move_legend(g, bbox_to_anchor=(5/6, 1/4*0.95), loc="center", title=None, fontsize=16)

    for lh in g.legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.fig.suptitle(title_label)

    filepath = os.path.join(save_folder, filename)
    plt.savefig(filepath)
    print('Saved:', filepath)

def plot_costs_categorised_stacked(data, filename, save_folder=desktop_folder, best_estimate=True, same_axes=True, is_diff=False, **kwargs):

    df = dcp(data)

    if is_diff:
        raise NotImplementedError('no stacked diffs')
        title_label = 'Difference in costs from baseline (2026-2030 inclusive)'
    else:
        title_label = 'Estimates of intervention costs (2026-2030 inclusive)'

    # Summarise costs
    df = df.groupby(['scen_name', 'estimate', 'cost_cat', 'node_long', 'scen_long'], sort=False)['cost'].sum().reset_index()

    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=df['scen_long'].unique())

    # Only plot "Intervention" costs
    df = df.loc[df['cost_cat'] == 'Intervention'].reset_index(drop=True)

    # Split data
    if best_estimate:
        df_plot = df.loc[df['estimate'] == 'best'].reset_index(drop=True)
    else:
        raise NotImplementedError('no stacked uncertainty')
        df_plot = df.loc[df['estimate'] != 'best'].reset_index(drop=True)

    df_plot = df_plot.drop(['estimate','cost_cat', 'scen_name'], axis='columns')

    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 2000)
    # print(df_plot)

    #
    # df_plot = df_plot.set_index(['scen_long']).pivot_table(values='cost', index='scen_long', columns='node_long', sort=False, observed=False)
    #
    # ax = df_plot.plot.bar(stacked=True)
    #
    # plt.title(title_label)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    # plt.tick_params(labelbottom=False)
    #
    # format_y_axes_label_costs_single(ax)
    # plt.subplots_adjust(right=0.8)
    # # g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    # # g.fig.suptitle(title_label)

    from seaborn import objects as so
    # plt.figure()

    p = (
        so.Plot(df_plot, x='scen_long', y='cost', color='node_long')
            .add(so.Bar(), so.Stack())
            .label(title=title_label, x=None, y='Total costs (2026-2030 inclusive)', legend='Intervention')
            .scale(color=so.Nominal(order=list(reversed(df_plot['node_long'].unique()))) )  # legend same order as stacked bars
         )

    fig, ax = plt.subplots()  # don't know if these two lines are the "right" way to get the so.Plot onto an ax but it works
    p.on(ax).plot()

    fig.subplots_adjust(bottom=0.1)  # Create space at bottom for scenario names
    plt.xticks(rotation=90, fontsize=10)

    format_y_axes_label_costs_single(ax)
    fig.legends[0].set_title("Intervention")
    fig.legends[0].set_loc('center left')
    fig.legends[0].set_bbox_to_anchor((1.05, 0.5))
    # sns.move_legend(fig, loc='center left', bbox_to_anchor=(1, 0.5), title=None, fontsize=16)
    # plt.tight_layout()


    filepath = os.path.join(save_folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print('Saved:', filepath)

def plot_costs_stacked(data, save_folder=desktop_folder):
    df = dcp(data)

    # plot_nodes = df['nodekey'].unique()
    plot_nodes = df['nodekey'].unique()

    df = df.loc[df['nodekey'].isin(plot_nodes)].reset_index(drop=True)
    df = df.drop(['outcome', 'value'], axis=1)

    value_columns = ['cost']
    df = df.groupby([col for col in df.columns if not col in value_columns], sort=False, dropna=False)[value_columns].sum().reset_index()

    cost_nodes = df['node_long'].unique()
    df = df.loc[df['node_long'].isin(cost_nodes)].reset_index(drop=True)
    df = df.loc[df['cost'] > 0].reset_index(drop=True)

    df['scen_long'] = pd.Categorical(df['scen_long'], ordered=True, categories=df['scen_long'].unique())

    # order = df.index

    # split data
    df_best = df.loc[df['estimate'] == 'best'].reset_index(drop=True)

    # df_best = df_best.sort_values(['dag_name', 'node_long']).reset_index(drop=True)

    # best estimate plot
    # plt.figure()
    sns.set_theme()
    g = sns.FacetGrid(
        data=df_best,
        hue='node_long',
        col='dag_name',
        col_order=list(df['dag_name'].unique()),
        col_wrap=2,
        legend_out=False,
        sharey=True,
    )
    g = (g.map(sns.barplot, data=df_best, x='scen_long', y='cost',
               order = df['scen_long'].cat.categories,
               errorbar=None).add_legend())
               # errorbar=None))

    g.set_titles('{col_name}')
    # g.set_axis_labels('Scenario', 'Total costs (2026-2030 inclusive) [$AU]')
    g.set(ylabel='Total costs (2026-2030 inclusive) [$AU]')
    g.set(xlabel=None)
    g.set(xticklabels=[])
    g.set(ylim=(0, None))

    # sns.move_legend(g, bbox_to_anchor=(5/6, 1/6), loc="center", title=None, fontsize=4)

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.90)  # adjust the Figure in rp
    g.fig.suptitle('Estimates of costs (2026-2030 inclusive)')

    filepath = os.path.join(save_folder, f'fig_costs_stacked_{TODAY_STR}.png')
    plt.savefig(filepath)
    print('Saved:', filepath)

def main_plotting(P, show_plots=False, sens_plot_kind=SENS_PLOT_KIND, image_type='png', save_folder=desktop_folder):
    if not show_plots: plt.switch_backend('agg')

    df = prepare_df_for_plotting(P.results)
    df_diff = prepare_df_for_plotting(P.results, P.results.df_diff)

    sc.makefilepath(folder=save_folder, makedirs=True)

    options = dict(sens_plot_kind=sens_plot_kind, save_folder=save_folder)

    plot_outcomes(df, filename=f'fig_outcomes_{TODAY_STR}.{image_type}', same_axes=False, **options)
    plot_outcomes(df, filename=f'fig_outcomes_sens_{TODAY_STR}.{image_type}', same_axes=False, best_estimate=False, **options)
    plot_outcomes(df_diff, filename=f'fig_outcomes_sens_diff_{TODAY_STR}.{image_type}', same_axes=False, best_estimate=False,
                  is_diff=True, **options)

    plot_outcomes_aggregated(df, filename=f'fig_outcomes_agg_{TODAY_STR}.{image_type}', same_axes=False, **options)
    plot_outcomes_aggregated(df_diff, filename=f'fig_outcomes_agg_diff_{TODAY_STR}.{image_type}', same_axes=False, is_diff=True, **options)
    plot_outcomes_aggregated(df, filename=f'fig_outcomes_agg_sens_{TODAY_STR}.{image_type}', same_axes=False,
                             best_estimate=False, **options)
    plot_outcomes_aggregated(df_diff, filename=f'fig_outcomes_agg_sens_diff_{TODAY_STR}.{image_type}', same_axes=False,
                             best_estimate=False, is_diff=True, **options)

    # plot_costs_best(df)
    plot_costs_categorised(df, filename=f'fig_costs_same_axes_{TODAY_STR}.{image_type}', same_axes=True, **options)
    plot_costs_categorised(df, filename=f'fig_costs_diff_axes_{TODAY_STR}.{image_type}', same_axes=False, **options)
    plot_costs_categorised(df_diff, filename=f'fig_costs_additional_same_axes_{TODAY_STR}.{image_type}', same_axes=True, is_diff=True, **options)
    plot_costs_categorised(df_diff, filename=f'fig_costs_additional_diff_axes_{TODAY_STR}.{image_type}', same_axes=False, is_diff=True, **options)
    # # plot_costs_stacked(df)

    plot_costs_categorised(df, filename=f'fig_costs_sens_same_axes_{TODAY_STR}.{image_type}', same_axes=True, best_estimate=False, **options)
    plot_costs_categorised(df, filename=f'fig_costs_sens_diff_axes_{TODAY_STR}.{image_type}', same_axes=False, best_estimate=False, **options)
    plot_costs_categorised(df_diff, filename=f'fig_costs_sens_additional_same_axes_{TODAY_STR}.{image_type}', best_estimate=False, same_axes=True, is_diff=True, **options)
    plot_costs_categorised(df_diff, filename=f'fig_costs_sens_additional_diff_axes_{TODAY_STR}.{image_type}', best_estimate=False, same_axes=False, is_diff=True, **options)

    plot_costs_categorised_stacked(df, filename=f'fig_costs_intervent_stacked_{TODAY_STR}.{image_type}', same_axes=False, **options)

    if show_plots: plt.show()

if __name__ == '__main__':

    show_plots = False

    # Get latest prj file sorted by !modified time!
    proj_file = sorted(sc.glob(folder=desktop_folder, pattern='**/*.prj', recursive=True), key=os.path.getmtime)[-1]
    print('Loading project:', proj_file)
    P = sc.load(filename=proj_file)

    save_folder = os.path.dirname(proj_file) + f'/Plots_{TIME_STR}/'  # Save plots into subfolder near proj_file

    main_plotting(P, show_plots=show_plots, save_folder=save_folder, image_type='png')
    main_plotting(P, show_plots=show_plots, save_folder=save_folder, image_type='pdf')

    print('Done')

