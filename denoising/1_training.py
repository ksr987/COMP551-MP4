#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import argparse

def main():
    args = parse_args()

    # Load and parse training data
    (X,Y), (X_val,Y_val), axes = load_training_data(
        args.train_data, validation_split=args.valid_split, axes=args.axes,
        verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    if args.plot:
        plt.figure(figsize=(12,5))
        plot_some(X_val[:5],Y_val[:5])
        plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
        plt.show()

    # Model config
    config = Config(
        axes, n_channel_in, n_channel_out, probabilistic = args.prob,
        train_steps_per_epoch = args.steps, train_epochs = args.epochs)
    print(vars(config))

    # Model init
    model = CARE(config, args.model_name, basedir='models')

    # Training, tensorboard available
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    # Plot training results
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    plt.savefig(args.model_name +'_training.png')
    plt.show()

    # Export model to be used w/ csbdeep fiji plugins and KNIME flows
    model.export_TF()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('train_data', type=str, help='training data to load')
    p.add_argument('--axes', type=str, default='SCZYX',
                   help='axes arg for load_training_data (def: SCZYX)')
    p.add_argument('model_name', type=str, help='name to save model to')
    p.add_argument('-e', '--epochs', type=int, default=100,
                   help='N training epochs (def: 100)')
    p.add_argument('-s', '--steps', type=int, default=400,
                   help='Steps per epoch (def: 400)')
    p.add_argument('--valid-split', type=float, default=0.2,
                   help='Porportion of training data to use for validation (def: 0.2)')
    p.add_argument('--probabilistic', action='store_true', dest='prob')
    p.add_argument('--non-probabilistic', action='store_false', dest='prob')
    p.add_argument('--no-plot', action='store_false', dest='plot')
    p.set_defaults(prob=True, plot=True)
    return p.parse_args()

if __name__ == '__main__':
    main()
