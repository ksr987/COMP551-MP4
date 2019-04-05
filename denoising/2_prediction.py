#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.models import CARE
from csbdeep.utils import plot_some
from csbdeep.utils import Path, download_and_extract_zip_file 
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
import argparse
from random import randint
from skimage.measure import compare_ssim as ssim
import util
import pandas as pd
import seaborn as sns
from csbdeep.data import Normalizer, PercentileNormalizer

# Example shows how to make predictions on the validation data used during
# training, not the test datasets as in the paper. Will need to set up more for
# that

def main():
    args = parse_args()

    X_val,Y_val = load_training_data(
        args.train_data, validation_split=args.valid_split, axes=args.axes,
        verbose=True)[1]
    
    # Config is saved during training and automatically loaded when init a CARE
    # with config=None
    model = CARE(config=None, name=args.model, basedir='models')

    if args.plot_random:
        plot_random_examples(X_val, Y_val, model, args.n_images)

    # too lazy to add a switch for this atm
    compute_metrics(X_val, Y_val, model, args.model)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=str, help='model to load (from basedir "models"')
    p.add_argument('train_data', type=str, 
                   help='training data to load validation images from')
    p.add_argument('--plot-random', type=bool, default=False,
                   help='Plot random training examples (def: False)')
    p.add_argument('--valid-split', type=float, default=0.2,
                   help='Porportion of training data to use for validation'
                   ', should be same as used during training (def: 0.2)')
    p.add_argument('-n', '--n-images', type=int, default=5,
                   help='The number of validation examples to plot def: 5')
    p.add_argument('--axes', type=str, default='SCZYX',
                   help='axes arg for load_training_data (def: SCZYX)')
    return p.parse_args()

def plot_random_examples(X_val, Y_val, model, n_images,
                         normalizer=PercentileNormalizer()):
    # Generate random indexes for plotting n traning examples
    ixs = get_randint_ixs(n_images, len(Y_val)-1)

    # Plot means (predictions)
    for ix in ixs:
        y = Y_val[ix,...,0]
        x = X_val[ix,...,0]
        axes = 'ZYX'

        restored = model.predict_probabilistic(x, axes, normalizer=normalizer)

        ims = [[x, restored.mean(), y]]
        titles = [['input '+str(ix), 'prediction', 'target']]
        plt.figure(figsize=(16,10))
        plot_some(np.stack(ims), title_list=titles)

    # Plot a line profile
    for ix in ixs:
        y = Y_val[ix,...,0]
        x = X_val[ix,...,0]
        axes = 'ZYX'

        restored = model.predict_probabilistic(x, axes, normalizer=None)

        i = 61
        line = restored[1, i, :]
        n = len(line)

        plt.figure(figsize=(16,9))
        plt.subplot(211)
        plt.imshow(restored.mean()[1, i-15:i+15, :], cmap='magma')
        plt.plot(range(n),15*np.ones(n),'--w',linewidth=2)
        plt.title('expected restored image %d' %ix)
        plt.xlim(0,n-1); plt.axis('off')

        plt.subplot(212)
        q = 0.025
        plt.fill_between(range(n), line.ppf(q), line.ppf(1-q), alpha=0.5,
                         label='%.0f%% credible interval'%(100*(1-2*q)))
        plt.plot(line.mean(),linewidth=3, label='expected restored image')
        plt.plot(y[1, i, :],'--',linewidth=3, label='ground truth')
        plt.plot(x[1, i, :],':',linewidth=1, label='input image')
        plt.title('line profile %d' %ix)
        plt.xlim(0,n-1); plt.legend(loc='lower right')
        plt.show()

def get_randint_ixs(n_ixs, max_ix, min_ix=0):
    ixs = []
    for i in range(n_ixs):
        while True:
            x = randint(min_ix, max_ix)
            if x not in ixs:
                ixs.append(x)
                break
    return ixs

def compute_metrics(X, Y, model, model_name='model', plot=True):
    d_pred = {'ssim':[], 'rmse': [], 'nrmse': []}
    d_input = {'ssim':[], 'rmse': [], 'nrmse': []}

    for ix in range(len(Y)):
        restored = util.get_prediction(model, X, ix, PercentileNormalizer())

        # x,y are float32, restored im is float64
        x = np.array(X[ix,...,0], dtype='float64')
        y = np.array(Y[ix,...,0], dtype='float64')
        pred = restored.mean()
        
        d_pred['ssim'].append(ssim(y, pred))
        d_input['ssim'].append(ssim(y, x))

        d_pred['rmse'].append(util.rmse_3d(y, pred))
        d_input['rmse'].append(util.rmse_3d(y, x))

        d_pred['nrmse'].append(util.nrmse_3d(y, pred))
        d_input['nrmse'].append(util.nrmse_3d(y, x))

    # Some jank longform dataframe creation
    _d = {'pred':d_pred, 'input':d_input}
    df_ls = []
    for k, v in _d.items():
        df = pd.DataFrame(v)
        df['im'] = range(len(df))
        df['output'] = [k for i in range(len(df))]
        df_ls.append(df)

    df_ls = (x for x in df_ls) # I don't know how to append to a (sequence)
    df = pd.concat(df_ls)
    df = pd.melt(df, id_vars=['output', 'im'], var_name='metric')

    if plot:
        g = sns.catplot(x='metric', y='value', hue='output', data=df, kind='bar')
        g.savefig(model_name +'_metrics.png')

    # Export summary statistics
    summary = { 'mean':{}, 'std':{} }
    metrics = df['metric']
    for o in df['output'].unique():
        summary['mean'][o] = {}
        summary['std'][o]  = {}
        for m in metrics.unique():
            c = [x for x in df.columns if x != 'im']
            tmp = df[c]
            tmp = tmp[tmp['output'] == o]
            tmp = tmp[tmp['metric'] == m]
            summary['mean'][o][m] = tmp.mean().value
            summary['std'][o][m] = tmp.std().value

    # Nested list
    df = pd.DataFrame.from_dict({(i,j): summary[i][j] 
                            for i in summary.keys() 
                            for j in summary[i].keys()},
                        orient='index')
    df.to_csv(model_name+'_metrics.csv')

if __name__ == '__main__':
    main()
