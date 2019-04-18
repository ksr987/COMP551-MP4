#!/usr/bin/env python
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import argparse
from six import string_types

def main(args):
    import json

    # Load and parse training data
    (X,Y), (X_val,Y_val), axes = load_training_data(
        args.train_data, validation_split=args.valid_split, axes=args.axes,
        verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


    # Model config
    print('args.resume: ', args.resume)
    if args.resume:
        # If resuming, config=None will reload the saved config
        config=None
        print('Attempting to resume')
    elif args.config:
        print('loading config from args')
        config_args = json.load(open(args.config))
        config = Config(**config_args)
    else:
        config = Config(
            axes, n_channel_in, n_channel_out, probabilistic = args.prob,
            train_steps_per_epoch = args.steps, train_epochs = args.epochs)
        print(vars(config))

    # Load or init model
    model = CARE(config, args.model_name, basedir='models')

    # Training, tensorboard available
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    # Plot training results
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    plt.savefig(args.model_name +'_training.png')

    # Export model to be used w/ csbdeep fiji plugins and KNIME flows
    model.export_TF()

def dev(args):
    import json

    # Load and parse training data
    (X,Y), (X_val,Y_val), axes = load_training_data(
        args.train_data, validation_split=args.valid_split, axes=args.axes,
        verbose=True)

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


    # Model config
    print('args.resume: ', args.resume)
    if args.resume:
        # If resuming, config=None will reload the saved config
        config=None
        print('Attempting to resume')
    elif args.config:
        print('loading config from args')
        config_args = json.load(open(args.config))
        config = Config(**config_args)
    else:
        config = Config(
            axes, n_channel_in, n_channel_out, probabilistic = args.prob,
            train_steps_per_epoch = args.steps, train_epochs = args.epochs)
        print(vars(config))

    # Load or init model
    model = CARE(config, args.model_name, basedir='models')

    # Training, tensorboard available
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    # Plot training results
    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16,5))
    plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
    plt.savefig(args.model_name +'_training.png')

    # Export model to be used w/ csbdeep fiji plugins and KNIME flows
    model.export_TF()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('train_data', type=str, help='training data to load')
    p.add_argument('model_name', type=str, help='name to save model to')
    p.add_argument('--axes', type=str, default='SCZYX',
                   help='axes arg for load_training_data (def: SCZYX)')
    p.add_argument('-e', '--epochs', type=int, default=100,
                   help='N training epochs (def: 100)')
    p.add_argument('-s', '--steps', type=int, default=400,
                   help='Steps per epoch (def: 400)')
    p.add_argument('--valid-split', type=float, default=0.2,
                   help='Porportion of training data to use for validation (def: 0.2)')
    p.add_argument('--probabilistic', action='store_true', dest='prob')
    p.add_argument('--non-probabilistic', action='store_false', dest='prob')
    p.add_argument('--resume', action='store_true', dest='resume', 
                   help='Load weights from model_name dir')
    p.add_argument('--config', type=str, 
                   help='load config options from this json file')
    p.add_argument('--dev', action='store_false', dest='main',
                   help='Run dev() instead of main()')
    p.set_defaults(prob=True, plot=True, resume=False, main=True)
    return p.parse_args()

def plot_history(history,*keys,**kwargs):
    """Plot (Keras) training history returned by :func:`CARE.train`.
    Changed to get rid of plt.show (which would hang up batch training
    executions)
    """

    logy = kwargs.pop('logy',False)

    if all(( isinstance(k,string_types) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,string_types) else group):
            plt.plot(history.epoch,history.history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')

if __name__ == '__main__':
    args = parse_args()
    if args.main:
        main(args)
    else:
        dev(args)
