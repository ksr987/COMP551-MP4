from csbdeep.io import load_training_data
from random import randint
from csbdeep.models import CARE
from csbdeep.data import Normalizer, PercentileNormalizer
from csbdeep.utils import plot_some
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_nrmse as nrmse
import numpy as np
import matplotlib.pyplot as plt

def load_data(npz_data, valid_only=True, valid_split=0.2, axes='SCZXY'):

    if valid_only:
        X_val,Y_val = load_training_data(
            npz_data, validation_split=valid_split, axes=axes,
            verbose=True)[1]
        return X_val, Y_val
    else:
        (X,Y), (X_val,Y_val), axes = load_training_data(
            npz_data, validation_split=valid_split, axes=axes,
            verbose=True)
        return (X,Y), (X_val,Y_val), axes

def load_data_and_model(npz_data, model_name, valid_split=0.2, axes='SCZXY'):

    model = CARE(config=None, name=model_name, basedir='models')
    X_val,Y_val = load_training_data(
        npz_data, validation_split=valid_split, axes=axes,
        verbose=True)[1]
    return X_val, Y_val, model

def get_randint_ixs(n_ixs, max_ix, min_ix=0):
    ixs = []
    for i in range(n_ixs):
        while True:
            x = randint(min_ix, max_ix)
            if x not in ixs:
                ixs.append(x)
                break
    return ixs

def get_prediction(model, X, ix, normalizer=PercentileNormalizer()):
    axes='ZYX'
    x = X[ix,...,0]
    return model.predict_probabilistic(x, axes, normalizer)

def plot_three(X, restored, Y, ix=None, use_ix=True, figsize=(16,10)):
    if use_ix:
        x = X[ix,...,0]
        y = Y[ix,...,0]
        titles = [['input %d' %ix, 'prediction', 'target']]
    else:
        x = X
        y = Y
        titles = [['input', 'prediction', 'target']]
    ims = [[x, restored.mean(), y]]
    plt.figure(figsize=figsize)
    plot_some(np.stack(ims), title_list=titles)

def rmse(y_true, y_pred):
    return(np.sqrt(mse(y_true, y_pred)))

def mse_3d(y_true, y_pred):
    """Get the mean mse of a 3 dim array with shape axes ZXY or ZYX"""
    z = []
    for i in range(len(y_true)):
        z.append(mse(y_true[i], y_pred[i]))
    return np.mean(z)

def rmse_3d(y_true, y_pred):
    """Get the mean rmse of a 3 dim array with shape axes ZXY or ZYX"""
    z = []
    for i in range(len(y_true)):
        z.append(rmse(y_true[i], y_pred[i]))
    return np.mean(z)

def nrmse_3d(y_true, y_pred, norm_type='min-max'):
    """Get the mean rmse of a 3 dim array with shape axes ZXY or ZYX
    I think the authors are using min-max"""
    z = []
    for i in range(len(y_true)):
        z.append(nrmse(y_true, y_pred, norm_type))
    return np.mean(z)
