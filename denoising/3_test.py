#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import util
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.models import CARE
from csbdeep.utils import plot_some
from tifffile import imread
import os
from os.path import join as pj
from os.path import split as ps
from csbdeep.data import Normalizer, PercentileNormalizer
import pandas as pd
import seaborn as sns
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse

def main():

    # The are the models to compare
    models = ['2019-04-01-planaria-1', '2019-04-03-planaria-3',
              '2019-04-02-planaria-4']

    # The types of catplots to make
    plot_kinds = ['bar', 'box', 'violin', 'swarm']

    # Dir to save results to
    outdir = 'test_results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ddir = 'data/Denoising_Planaria/test_data'
    gt_dir = pj(ddir, 'GT')
    conditions = [pj(ddir, '_'.join(('condition', str(i)))) for i in range(1,4)]


    df_ls = []
    for c in conditions:
        print('\n%s'%c)
        d = run_multi(gt_dir, c, models)

        # Costruct a df for the barplot and add a condition col
        df = pd.DataFrame(d['output'], columns=d['columns'], )
        df = pd.melt(df, id_vars=d['id_vars'], var_name=d['var_name'])
        df['condition'] = os.path.basename(c)
        df_ls.append(df)

    # Output the longform raw data
    df = pd.concat(df_ls) # Casually overwrite df for maximum confusion
    df.to_csv(pj(outdir,'test_raw.csv'), index=False)

    # Output mean/std
    summary = df.groupby(['condition', 'output', 'metric']).describe()
    # drop unwanted 'value' multiindex from describe
    summary.columns = summary.columns.droplevel() 
    summary = summary[['mean', 'std']]
    summary.to_csv(pj(outdir,'test_summary.csv'))

    # Plots
    plt.figure(figsize=(16,10))
    for kind in plot_kinds:
        g = sns.catplot(x='condition', y='value', hue='output', kind=kind,
                        col='metric', sharey=False, data=df)

        # Set SSIM ylim to 1, since 1 is best symetery. For (N)RMSE, let it be
        # set by data, since best is RMSE -> 0
        g.axes[0,1].set_ylim(0,1)
        plt.savefig(pj(outdir, 'test_%s.png'%kind))
        # plt.show();

def run_multi(ydir, xdir, models):

    # Prepare the dict for metrics. Half of these don't even really get used, I
    # think
    d = {
        'im_name'  : [],
        'im_ix'    : [],
        'output'   : [],
        'columns'  : ['output', 'rmse', 'ssim'],
        'id_vars'  : ['output'],
        'var_name' : 'metric'
    }

    gt = os.listdir(ydir)
    gt.sort()
    y_files = [pj(ydir, x) for x in gt]
    inputs = os.listdir(xdir)
    inputs.sort()
    x_files = [pj(xdir, x) for x in inputs]

    d = _run_multi(y_files, x_files, models, d)
    return d

def _run_multi(y_files, x_files, models, d):

    # Define comparisons
    def get_output(name, y, x):
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    for x, y in zip(y_files, x_files):
        assert os.path.basename(x) == os.path.basename(y)
        # Prep data
        # model prediction and normalizations output float64
        x = imread(x)
        y = imread(y)
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='float64')
        axes = 'ZYX'

        # Normalize GT dynamic range to enable comparisons w/ numbers
        yn = util.percentile_norm(y, axes)
        xn = util.percentile_norm(x, axes)

        # Get the comparison for normalized input im
        d['output'].append(get_output('input', yn, xn))

        for m in models:
            print('Model: %s'%m)
            model = CARE(config=None, name=m, basedir='models')

            # Make prediction
            restored = model.predict_probabilistic(x, axes, n_tiles=(1,4,4))
            pred = restored.mean()
            y_pred_n = util.percentile_norm(pred, axes)
            d['output'].append(get_output(m, yn, y_pred_n))

    return d 

if __name__ == '__main__':
    main()
