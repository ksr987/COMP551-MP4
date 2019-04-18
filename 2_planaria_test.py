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
from skimage.restoration import denoise_nl_means, estimate_sigma
import datetime
import argparse
from csbdeep.utils import normalize, normalize_minmse

def run_z(y, x, models):
    # Prep data
    # model prediction and normalizations output float64
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    axes = 'ZYX'

    # Prepare the dict for metrics
    d = {
        'output'  : [],
        'columns' : ['output', 'rmse', 'ssim'],
        'id_vars' : ['output'], 
        'var_name' : 'metric'
    }

    # Define comparisons
    def get_output(name, y, x):
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    # Normalize GT dynamic range to enable comparisons w/ numbers
    yn = util.percentile_norm(y, axes)

    # Get the comparison for normalized input im
    xn = util.percentile_norm(x, axes)
    d['output'].append(get_output('input', yn, xn))

    predictions = {'models':['input', 'N(GT)'], 'ims':[xn, yn]}
    for m in models:
        model = CARE(config=None, name=m, basedir='models')
        restored = model.predict_probabilistic(x, axes, n_tiles=(1,4,4))
        pred = restored.mean()
        y_pred_n = util.percentile_norm(pred, axes)

        d['output'].append(get_output(m, yn, y_pred_n))
        predictions['models'].append(m)
        predictions['ims'].append(y_pred_n)
     
    # Plot a random stack
    zix = util.get_randint_ixs(1, len(y))
    ims = [[im[zix] for im in predictions['ims']]]
    plt.figure(figsize=(16,10))
    plot_some(np.stack(ims), title_list=[predictions['models']])
    plt.show();
    
    # Costruct a df for the barplot
    df = pd.DataFrame(d['output'], columns=d['columns'], )
    df = pd.melt(df, id_vars=d['id_vars'], var_name=d['var_name'])

    g = sns.catplot(x='metric', y='value', hue='output', kind='bar',
                    sharey=False,
                    data=df)
    g.ax.set_ylim(0,1)
    plt.show();

def main():

    args = parse_args()
    ddir = 'data/Denoising_Planaria/test_data'
    gt_dir = pj(ddir, 'GT')
    conditions = [pj(ddir, '_'.join(('condition', str(i)))) for i in range(1,4)]

    if args.load_models:
        models = [line.rstrip('\n') for line in open(args.load_models)]
    else:
        models = ['2019-04-06-beetle-1']

    # The types of catplots to make
    plot_kinds = ['bar', 'box']

    # Dir to save results to
    outdir = 'planaria_results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    df_ls = []
    for c in conditions:
        print('\n%s'%c)

        if args.network:
            d = run_multi(gt_dir, c, models)

            # Costruct a df for the barplot and add a condition col
            df = pd.DataFrame(d['output'], columns=d['columns'], )
            df = pd.melt(df, id_vars=d['id_vars'], var_name=d['var_name'])
            df['condition'] = os.path.basename(c)
            df_ls.append(df)

        if args.nlm:
            d = test_nlm(gt_dir, c)

            # Costruct a df for the barplot and add a condition col
            df = pd.DataFrame(d['output'], columns=d['columns'], )
            df = pd.melt(df, id_vars=d['id_vars'], var_name=d['var_name'])
            df['condition'] = os.path.basename(c)
            df_ls.append(df)

    # Casually overwrite df for maximum confusion
    df = pd.concat(df_ls)

    # Output the longform raw data
    df = pd.concat(df_ls) # Casually overwrite df for maximum confusion
    outfile = file_namer(outdir,'test_raw.csv')
    df.to_csv(outfile, index=False)

    # Output mean/std
    summary = df.groupby(['condition', 'output', 'metric']).describe()
    # drop unwanted 'value' multiindex from describe
    summary.columns = summary.columns.droplevel() 
    summary = summary[['mean', 'std']]
    outfile = file_namer(outdir,'test_summary.csv')
    summary.to_csv(outfile)

    plt.figure(figsize=(16,10))
    for kind in plot_kinds:
        g = sns.catplot(x='condition', y='value', hue='output', kind=kind,
                        col='metric', sharey=False, data=df)

        # Set SSIM ylim to 1, since 1 is best symetery. For (N)RMSE, let it be
        # set by data, since best is RMSE -> 0
        g.axes[0,1].set_ylim(0,1)
        plt.savefig(file_namer(outdir, '%s.png' % kind))

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
    inputs = os.listdir(xdir)
    inputs.sort()

    i = 0
    for x, y in zip(inputs, gt):
        assert x == y
        d['im_name'].append(x)
        d['im_ix'].append(i)  # unecessary?
        y = pj(ydir, y)
        x = pj(xdir, x)
        d = _run_multi(imread(y), imread(x), models, d)
    return d

def _run_multi(y, x, models, d):
    # Prep data
    axes = 'ZYX'
    # Define comparisons
    def get_output(name, y, x):
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    # Normalize GT dynamic range to enable comparisons w/ numbers
    yn = normalize(y, pmin=0.1, pmax=99.9)

    # Get the comparison for normalized input im
    xn = normalize(x)
    d['output'].append(get_output('input', yn, xn))

    for m in models:
        model = CARE(config=None, name=m, basedir='models')
        # None normalizer, already normalizing x, and this way can report the
        # exact pmin/max params used
        pred = model.predict(xn, n_tiles=(1,4,4), axes=axes, normalizer=None)
        pred = normalize_minmse(pred, yn)
        d['output'].append(get_output(m, yn, pred))
     
    return d 

def file_namer(dirpath, name_w_ext):
    def ifname_append(file_name):
        """ Append file name if it already exists
        """
        if not os.path.isfile(file_name):
            return file_name
        else:
            expand = 1
            while True:
                expand += 1
                split = os.path.splitext(file_name)
                new_file_name = split[0] +'-'+ str(expand) + split[-1]
                if os.path.isfile(new_file_name):
                    continue
                else:
                    return new_file_name

    date = datetime.date.today()
    outfile = os.path.join(dirpath, '%s-%s' %(date, name_w_ext))
    outfile = ifname_append(outfile)
    return outfile

def test_nlm(ydir, xdir):
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
    inputs = os.listdir(xdir)
    inputs.sort()

    i = 0
    for x, y in zip(inputs, gt):
        assert x == y
        d['im_name'].append(x)
        d['im_ix'].append(i)  # unecessary?
        y = pj(ydir, y)
        x = pj(xdir, x)
        d = _test_nlm(imread(y), imread(x), d)
    return d

def _test_nlm(y, x, d):
    # Prep data
    # model prediction and normalizations output float64
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    axes = 'ZYX'


    # Define comparisons
    def get_output(name, y, x):
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    # Normalize GT dynamic range to enable comparisons w/ numbers
    yn = util.percentile_norm(y, axes)

    # Get the comparison for normalized input im
    xn = util.percentile_norm(x, axes)
    d['output'].append(get_output('input', yn, xn))

    # Compute NLM
    patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,      # 13x13 search area
                multichannel=True)
    # estimate sigma from noisy data
    sigma_est = np.mean(estimate_sigma(x, multichannel=True))
    
    # slow algorithm, sigma provided
    denoise2 = denoise_nl_means(x, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)
    nlmn = util.percentile_norm(denoise2, axes)
    d['output'].append(get_output('nlm', yn, nlmn))
    return d

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--load-models', type=str)
    p.add_argument('--nlm', action='store_true', dest='nlm')
    p.add_argument('-nn', '--no-net', action='store_false', dest='network')
    p.set_defaults(network=True, nlm=False)
    return p.parse_args()

if __name__ == '__main__':
    main()
