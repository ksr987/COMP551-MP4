#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import util
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.models import CARE
from csbdeep.utils import plot_some
from csbdeep.data import Normalizer, PercentileNormalizer
from csbdeep.utils import normalize, normalize_minmse
from tifffile import imread
import os
from os.path import join as pj
from os.path import split as ps
import pandas as pd
import seaborn as sns
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import datetime
import argparse
from skimage.restoration import denoise_nl_means, estimate_sigma

def main():
    args = parse_args()

    # The are the models to compare
    if args.load_models:
        models = [line.rstrip('\n') for line in open(args.load_models)]
    else:
        models = ['2019-04-06-beetle-1']

    # The types of catplots to make
    plot_kinds = ['bar', 'box']

    # Dir to save results to
    outdir = 'beetle_results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load a dict w/ keys condition<1-3> and y, y is GT
    test_dir = 'data/Denoising_Tribolium/test_data'
    x_conds, y = load_test_data(test_dir)

    df_ls = []
    for condition, x in x_conds.items():
        print('\n%s'%condition)
        d = run_multi(y, x, models, condition, args)

        # Costruct a df for the barplot and add a condition col
        df = pd.DataFrame(d['output'], columns=d['columns'], )
        df = pd.melt(df, id_vars=d['id_vars'], var_name=d['var_name'])
        df['condition'] = os.path.basename(condition)
        df_ls.append(df)

    # Casually overwrite df for maximum confusion
    df = pd.concat(df_ls)

    # Output the longform raw data
    df = pd.concat(df_ls) # Casually overwrite df for maximum confusion
    outfile = file_namer(outdir, 'test_raw.csv')
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

        if args.alt_norm:
            g.title('PercentileNormalizer(do_after=False)')
        plt.savefig(file_namer(outdir, '%s.png' % kind))

def run_multi(X, Y, models, condition, args):

    # Prepare the dict for metrics. Half of these don't even really get used, I
    # think
    d = {
        'output'   : [],
        'condition': condition,
        'columns'  : ['output', 'rmse', 'ssim'],
        'id_vars'  : ['output'],
        'var_name' : 'metric'
    }

    ix = 0
    for x, y in zip(X, Y):
        if args.nlm:
            _test_nlm(y, x, d)
        elif args.rescale_x:
            d = _run_input(y, x, models, d, ix, args)
        else:
            d = _run_multi(y, x, models, d, ix, args)
        ix += 1
    return d

def _run_multi(y, x, models, d, ix, args):
    """ Trying an alternative normalization to see if that's causing the
    incong. with the paper results
    """
    # Prep data
    axes = 'ZYX'

    def get_output(name, y, x):
        # Define comparisons
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    # Normalize GT dynamic range to enable comparisons w/ numbers
    yn = normalize(y, pmin=0.1, pmax=99.9)

    # Get the comparison for normalized input im
    if args.rescale_x:
        xn = normalize_minmse(x, yn)
    else:
        xn = normalize(x)
    d['output'].append(get_output('input', yn, xn))

    for m in models:
        model = CARE(config=None, name=m, basedir='models')
        pred = model.predict(xn, axes, n_tiles=(1,4,4), normalizer=None)
        pred = normalize_minmse(pred, yn)
        d['output'].append(get_output(m, yn, pred))

        # Save the first test volume for each model
        if args.save_predictions:
            if ix == 0:
                save_prediction(pred, m, d['condition'])

    # Save the first test volume input and GT
    if args.save_predictions:
        if ix == 0:
            save_prediction(xn, 'input', d['condition'])
            save_prediction(yn, None, d['condition'])
     
    return d 

def _run_input(y, x, models, d, ix, args):
    """ Trying an alternative normalization to see if that's causing the
    incong. with the paper results
    """
    # Prep data
    axes = 'ZYX'

    def get_output(name, y, x):
        # Define comparisons
        return [name, np.sqrt(mse(x, y)), ssim(x, y)]

    # Normalize GT dynamic range to enable comparisons w/ numbers
    yn = normalize(y, pmin=0.1, pmax=99.9)

    # Get the comparison for normalized input im
    if args.rescale_x:
        xn = normalize_minmse(x, yn)
    else:
        xn = normalize(x)
    d['output'].append(get_output('input', yn, xn))

    return d 

def load_test_data(test_dir):
    """ Return dict w/ keys condition 1--3 and y. 
    Each value is a list of samples. Each sample is a np 3darray z stack of one
    sample, axes ZYX
    """
    im_files = os.listdir(test_dir)
    data = {'condition_%d' %(i+1) : [] for i in range(3)}
    y = []
    im_names = []

    for f in im_files:
        im = imread(pj(test_dir, f))
        for i in range(3):
            data['condition_%d' %(i+1)].append(im[i])
        y.append(im[-1])

    return data, y

def plot_test_data(test_dir):
    """ Plot a grid with im stacks as rows, conditions as columns
    """

    test_dir = 'data/Denoising_Tribolium/test_data'
    im_files = os.listdir(test_dir)

    data = {'condition_%d' %(i+1) : [] for i in range(3)}
    data['y'] = []
    im_names = []

    for f in im_files:
        im_names.append(f)
        im = imread(pj(test_dir, f))
        _shape = im.shape
        print(im.shape)
        
        for i in range(3):
            data['condition_%d' %(i+1)].append(im[i])
        data['y'].append(im[-1])

    plt.rcParams['figure.figsize'] = [18, 6]
    titles = ['condition_%d' %(i+1) for i in range(3)]
    titles.append('y (GT)')

    for j in range(len(im_names)):
        i = 0
        f, ax = plt.subplots(1,4)
        for k, v in data.items():
            ax[i].imshow(v[j][0])
            ax[i].set_title(titles[i])
            i = i + 1
        f.suptitle(os.path.basename(im_names[i]))

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

def save_prediction(x, model, condition=None, save_dir='predictions/beetle_test_data'):
    # My method of saving the arrays is jank, but TBH I'm really fatigue

    # Where to save the prediction
    if model:
        outdir = pj(save_dir, model, condition)
    else:
        # model=None, Assume this is GT
        outdir = save_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    im_name = 'beetle_test_volume_1'
    outfile = pj(outdir, im_name+'.npy')
    np.save(outfile, x)
    print('Saved %s'%outfile)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--load-models', type=str)
    p.add_argument('--save-predictions', action='store_true',
                   dest='save_predictions',
                   help='Save an example volume for each model/condition')
    p.add_argument('--alt-norm', action='store_true', dest='alt_norm',
                   help='Use alternative normalizaton')
    p.add_argument('--nlm', action='store_true', dest='nlm',
                  help='Just compute nlm (estimated sigma) for this set')
    p.add_argument('--rescale-x', action='store_true', dest='rescale_x',
                   help='affine rescale x before computing rmse and ssim')
    p.set_defaults(nlm=False, save_predictions=False, rescale_x=False)
    return p.parse_args()


if __name__ == '__main__':
    main()
