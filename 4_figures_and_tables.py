#!/usr/bin/env python
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import join as pj
from os.path import split as ps
from tifffile import imread
from skimage.restoration import denoise_nl_means, estimate_sigma
from csbdeep.data import Normalizer, PercentileNormalizer
from glob import glob
import re

def rename_output(df, new, old='planaria-1'):
    df.loc[df['output'] == old, 'output'] = new
    return df

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def _consolidate_tables():
    def consolidate_tables(glob_str, outfile):
        df_ls = [pd.read_csv(f) for f in glob(glob_str)]
        df_ls = [pd.concat(df_ls, ignore_index=True).drop_duplicates()]
        
        if 'summary' in glob_str:
            df_ls.append(
                df_ls.append(pd.read_csv(
                    pj(os.path.dirname(glob(glob_str)[0]), 'paper_results.csv'))
                )
            )  
            for df, x in zip(df_ls, ['replication', 'original']):
                df['study'] = x 

        df = pd.concat(df_ls,  sort=True)
        # drop the dates form model names
        df['output'] = [re.split('\d{4}-\d{2}-\d{2}-', o)[-1] for o in df['output']]
        df.to_csv(outfile, index=False)


    outs = ['tables/planaria_test_data_raw.csv',
            'tables/planaria_test_data_summary.csv',
            'tables/beetle_test_data_raw.csv',
            'tables/beetle_test_data_summary.csv']
    ddirs = ['planaria_results/2019-04-16-test_raw*.csv',
            'planaria_results/2019-04-16-test_summary*.csv',
            'beetle_results/2019-04-16-test_raw.csv',
            'beetle_results/2019-04-16-test_summary.csv',
            ]

    for ddir, out in zip(ddirs, outs):
        consolidate_tables(ddir, out)

    # correct planaria-8 >> planaria-batch8
    for out in outs[:2]:
        df = pd.read_csv(out)
        df.loc[df['output'] == 'planaria-8', 'output'] = 'planaria-batch8'
        df.to_csv(out, index=False)


    def rename_output(df, new, old='planaria-1'):
        df.loc[df['output'] == old, 'output'] = new
        return df

def plots(save_dir='plots/final'):

    def _plots(df, cols, save_name, value='value', kind='bar', labs=None,
              aspect=16/10, height=3):
        g = sns.catplot(x='condition', y=value, 
                        hue='output', hue_order=cols,
                        col='metric',
                        kind=kind,
                        aspect=aspect, height=3,
                        sharey=False, data=df)
        g.axes[0,1].set_ylim(0,1)
        g.set_ylabels('')
        g.axes[0,0].set_title('RMSE')
        g.axes[0,1].set_title('SSIM')
        if labs is not None:
            for t, l in zip(g._legend.texts, labs): t.set_text(l)
        plt.savefig(pj(save_dir, '_'.join((save_name, '%s.png' % kind))))
        plt.show()

    def worms_hyper():
        df = pd.read_csv('tables/planaria_test_data_raw.csv')
        base = ['input']

        # batch
        save_name = 'hyper_batch_size'
        _df = df.copy()
        _df = rename_output(_df, 'planaria-batch16') 
        outputs = _df['output'].unique()
        cols = base + sorted_nicely([x for x in outputs if 'batch' in x])
        labs = [x.split('planaria-')[-1].title() for x in cols]
        _plots(_df, cols, save_name, labs=labs)

        # step
        save_name = 'hyper_steps_per_epoch'
        _df = df.copy()
        _df = rename_output(_df, 'planaria-steps400') 
        outputs = _df['output'].unique()
        cols = base + sorted_nicely([x for x in outputs if 'step' in x])
        labs = [x.split('planaria-')[-1].title() for x in cols]
        _plots(_df, cols, save_name, labs=labs)
    
    def generalization():
        # worms test data
        df = pd.read_csv('tables/planaria_test_data_raw.csv')
        base = ['input']

        df = rename_output(df, 'planaria-') 

    def rep():
        # worms
        df = pd.read_csv('tables/planaria_test_data_summary.csv')
        base = ['input']

        renames = [('paper_input', 'paper input'),
                   ('paper_network', 'paper network'),
                   ('planaria-1', '100 epochs'),
                   ('planaria-3', '200 epochs'),
                   ('planaria-batch64', '100 epochs, batch 64')
                  ]
        for tup in renames:
            rename_output(df, tup[1], tup[0])

        extra = ['paper input', 
                 '100 epochs', 
                 '200 epochs', 
                 '100 epochs, batch 64',
                 'paper network'
                ]

        cols = base + extra
        _plots(df, cols, 'replication_planaria', 'mean')

        # beetle
        df = pd.read_csv('tables/beetle_test_data_summary.csv')
        base = ['input']

        renames = [('paper_input', 'paper input'),
                   ('beetle-1', '100 epochs'),
                   ('beetle-2', '200 epochs'),
                   ('paper_network', 'paper network')
                  ]
        for tup in renames:
            rename_output(df, tup[1], tup[0])

        cols = base + [tup[-1] for tup in renames]
        _plots(df, cols, 'replication_beetle', 'mean')

    def generalize():
        # worm test data
        df = pd.read_csv('tables/planaria_test_data_raw.csv')
        base = ['input']
        print(df['output'].unique())

        # renames = [('paper_input', 'paper input'),
        #            ('paper_network', 'paper network'),
        #            ('planaria-1', '100 epochs'),
        #            ('planaria-3', '200 epochs'),
        #            ('planaria-batch64', '100 epochs, batch 64')
        #           ]

        cols = base + ['beetle-1', 'planaria-1']
        _plots(df, cols, 'generalize_planaria_data')
    
        # beetle test data
        df = pd.read_csv('tables/beetle_test_data_raw.csv')
        base = ['input']
        print(df['output'].unique())

        cols = base + ['beetle-1', 'planaria-1']
        _plots(df, cols, 'generalize_beetle_data')

        # in one plot
        fs = ['tables/planaria_test_data_raw.csv', 
              'tables/beetle_test_data_raw.csv']
        df_ls = [pd.read_csv(f) for f in fs]

        for df, d in zip(df_ls, ['planaria', 'beetle']):
            df['dataset'] = d
        df = pd.concat(df_ls, sort=True)

        kind = 'bar'
        save_name = 'generalization_test'
        g = sns.catplot(x='condition', y='value', 
                        hue='output', hue_order=cols,
                        col='metric',
                        row='dataset',
                        height=3, aspect = 16/10,
                        kind=kind,
                        sharey=False, data=df)
        for i, axes_row in enumerate(g.axes):
            for j, axes_col in enumerate(axes_row):
                row, col = axes_col.get_title().split('|')
                if j == 0:
                    axes_col.set_ylabel(row.title())

            g.axes[i,0].set_ylim(0,0.35)
            g.axes[i,1].set_ylim(0,1)
            g.axes[i,0].set_title('RMSE')
            g.axes[i,1].set_title('SSIM')
        # Relabel legend
        labs = ['Input', 'Tribolium', 'Planaria']
        for t, l in zip(g._legend.texts, labs): t.set_text(l)
        plt.savefig(pj(save_dir, '_'.join((save_name, '%s.png' % kind))))
        
    # Jank, but only needs to run once, right?
    os.makedirs(save_dir)

    # Run child functions
    worms_hyper()
    rep()
    generalize()

def tables(save_dir='tables/final'):
    def _tables(df, outs, save_name):
        df = df.drop('std', axis=1)
        df = df[df['output'].isin(outs)]
        df = df.pivot_table(index='condition', columns=['metric', 'output'],
                            values='mean')

        df.to_csv(pj(save_dir,save_name), float_format='%.5f')

    os.makedirs(save_dir)

    # Planaria replication
    df = pd.read_csv('tables/planaria_test_data_summary.csv')
    base = ['input']
    paper = ['paper_input', 'paper_network']
    extra = ['planaria-1']
    outs = base + extra + paper
    _tables(df, outs, 'planaria_replication.csv')
    # ours only
    _tables(df, outs, 'planaria_replication.csv')

    # Beetle replication
    df = pd.read_csv('tables/beetle_test_data_summary.csv')
    base = ['input']
    paper = ['paper_input', 'paper_network']
    extra = ['beetle-1']
    outs = base + extra + paper
    _tables(df, outs, 'beetle_replication.csv')

    # Batch
     


plots()
tables()
