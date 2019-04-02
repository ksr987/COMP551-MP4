#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import CARE
import argparse
from random import randint

# Example shows how to make predictions on the validation data used during
# training, not the test datasets as in the paper. Will need to set up more for
# that

def main():
    args = parse_args()

    # (X,Y), (X_val,Y_val), axes = load_training_data(
    #     args.train_data, validation_split=args.valid_split, axes=args.axes,
    #     verbose=True)
    X_val,Y_val = load_training_data(
        args.train_data, validation_split=args.valid_split, axes=args.axes,
        verbose=True)[1]
    
    # Config is saved during training and automatically loaded when init a CARE
    # with config=None
    model = CARE(config=None, name=args.model, basedir='models')

    # Generate random indexes for plotting n traning examples
    ixs = get_randint_ixs(args.n_images, len(Y_val)-1)

    # Plot means (predictions)
    for ix in ixs:
        y = Y_val[ix,...,0]
        x = X_val[ix,...,0]
        axes = 'ZYX'

        # None normalizer is probably wrong, but need to review the authors SI
        restored = model.predict_probabilistic(x, axes, normalizer=None)

        ims = [[x, restored.mean(), y]]
        titles = [['input '+str(ix), 'prediction', 'target']]
        plt.figure(figsize=(16,10))
        plot_some(np.stack(ims), title_list=titles)

    # Plot a line profile
    for ix in ixs:
        y = Y_val[ix,...,0]
        x = X_val[ix,...,0]
        axes = 'ZYX'

        # None normalizer is probably wrong, but need to review the authors SI
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=str, help='model to load (from basedir "models"')
    p.add_argument('train_data', type=str, 
                   help='training data to load validation images from')
    p.add_argument('--valid-split', type=float, default=0.2,
                   help='Porportion of training data to use for validation'
                   ', should be same as used during training (def: 0.2)')
    p.add_argument('-n', '--n-images', type=int, default=5,
                   help='The number of validation examples to plot def: 5')
    p.add_argument('--axes', type=str, default='SCZYX',
                   help='axes arg for load_training_data (def: SCZYX)')
    p.add_argument('--plot-scale', type=bool, default=False,
                   help='Plot pixelwise scale')
    # p.add_argument('--plot-var', type=bool, default=False,
    #                help='Plot pixelwise variance')
    # p.add_argument('--plot-entropy', type=bool, default=False,
    #                help='Plot pixelwise entropy')
    return p.parse_args()

def get_randint_ixs(n_ixs, max_ix, min_ix=0):
    ixs = []
    for i in range(n_ixs):
        while True:
            x = randint(min_ix, max_ix)
            if x not in ixs:
                ixs.append(x)
                break
    return ixs

if __name__ == '__main__':
    main()
