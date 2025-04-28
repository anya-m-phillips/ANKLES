import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.stats as stats
from torch.distributions import Distribution
from sklearn.neighbors import KernelDensity
import torch
import pandas as pd

from tqdm import tqdm
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
# from sbi.inference import load_posterior

from sklearn.model_selection import train_test_split

import seaborn as sns

import itertools
from matplotlib.lines import Line2D
import random

import astropy.constants as const
import astropy.units as u

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from torch.distributions import constraints

import corner

from priors import KDEPrior

## load min/max from training (for un-normalizing outputs!)
dat = np.load("models/training_extrema_SBI.npz")
min_tr = dat['min_tr']
max_tr = dat['max_tr']

def normalize_inputs(input_data):
    """
    input_data must be a numpy array of shape (N, 40)
    where the first 20 columns are RVs 
    and the second 20 columns are times of the observations.
    Returns (N, 42) array: 20 normalized RVs, 20 normalized times, v0_est, K_est
    """
    input_RVs = input_data[:, :20]
    input_times = input_data[:, 20:]

    # Compute RV normalizations
    v_max = np.max(input_RVs, axis=1, keepdims=True)
    v_min = np.min(input_RVs, axis=1, keepdims=True)
    v0_est = np.mean(input_RVs, axis=1, keepdims=True)
    K_est = (v_max - v_min) / 2

    # Normalize RVs
    normed_rvs = (input_RVs - v_min) / (v_max - v_min + 1e-8)  # add epsilon to avoid div by zero

    # Compute time normalizations
    t_max = np.max(input_times, axis=1, keepdims=True)
    t_min = np.min(input_times, axis=1, keepdims=True)

    # Normalize times
    normed_times = (input_times - t_min) / (t_max - t_min + 1e-8)

    return np.hstack([normed_rvs, normed_times, v0_est, K_est])    

def normalize_outputs(output_data, min_tr=min_tr, max_tr=max_tr):
    """
    normalize outputs based on the min/max of the training data
    normalized column-wise (ie so that all orbital periods are btwn 0--1, etc)
    """
    normed_outputs = (output_data - min_tr)/(max_tr - min_tr)
    return normed_outputs

def unnormalize_outputs(output_data, min_tr=min_tr, max_tr=max_tr):
    unnormed_outputs = (output_data*(max_tr - min_tr)) + min_tr
    return unnormed_outputs
    

# the machine learning part
def sample_posterior(input_data, n_samples, model_path="models/sbi_model_longtraining.pt"):
    """
    provide input_data as a numpy array of 20 RVs, 20 corresponding times
    """
    X_numpy = normalize_inputs(input_data)
    X = torch.tensor(X_numpy, dtype=torch.float32)


    ### CONSTRUCT PRIOR
    normed_labels_dat = np.load("data/normed_labels.npz")
    samples = normed_labels_dat['samples']
    samples = torch.tensor(samples, dtype=torch.float32)  # shape [[long], 6] ###
    prior = KDEPrior(samples)


    hatp_x_y = torch.load(model_path, weights_only=False) 

    samples = hatp_x_y.sample((n_samples,), x=X).numpy() 
    unnormed_samples = unnormalize_outputs(samples, min_tr=min_tr, max_tr=max_tr)

    return unnormed_samples


# stats functions
def stats(samples):
    """
    return median and 16th--84th percentile range of posterior samples
    """
    med = np.median(samples, axis=0)
    unc =  (np.percentile(samples,84, axis=0)-np.percentile(samples,16, axis=0))/2.0
    return med, unc


##### generating new RV curves??

    

def generate_orbital_params(N):
    """
    generate N sets of orbital parameters
    (based on prior)
    """
    ### CONSTRUCT PRIOR
    normed_labels_dat = np.load("data/normed_labels.npz")
    samples = normed_labels_dat['samples']
    samples = torch.tensor(samples, dtype=torch.float32)  # shape [[long], 6] ###
    prior = KDEPrior(samples)

    generated_params = prior.sample((N,))
    generated_params_unnorm = unnormalize_outputs(generated_params)
    return generated_params_unnorm

def obsEpochs(N):
    """
    generate [20] observation epochs for N RV curves
    """
    Nobs_g = np.array([3, 20])#np.arange(2, 6+1)

    e_rv_g = np.array([0.1, 1.0, 3.0])
    Ng = len(e_rv_g)

    Nobsm = np.max(Nobs_g)

    dt_min = 30 
    dt_max = 125

    rng = np.random.default_rng(seed = 42)

    deltaTs = 10**rng.uniform(np.log10(dt_min), np.log10(dt_max), size = (N, Nobsm-1))  ### a *log* uniform distribution in deltaTs
    #deltaTs = np.random.uniform(dt_min, dt_max, size = (Nstar, Nobsm - 1)) # N >= 2 visit s
    deltaTs = np.column_stack((np.zeros(N), deltaTs)) # add t = 0 visit

    obstimes_all = np.cumsum(deltaTs, axis = 1)
    return obstimes_all

## functions to solve keplers equation, get the true anomaly, and get RVs
def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """Vectorized Kepler solver using Newton-Raphson with safety."""
    M = np.asarray(M)
    e = np.asarray(e)

    # Ensure shapes are compatible
    if M.shape != e.shape:
        e = np.full_like(M, e)

    E = M.copy()  # initial guess

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime

        # Protect against divide-by-zero or overflow
        delta = np.where(np.isfinite(delta), delta, 0.0)
        E_new = E - delta

        # Convergence mask
        if np.all(np.abs(delta) < tol):
            break
        E = E_new

    # Final sanity check
    E = np.where(np.isfinite(E), E, np.nan)
    return E

def true_anomaly(E, e):
    """Convert eccentric anomaly E to true anomaly ν safely."""
    sqrt_1_plus_e = np.sqrt(np.clip(1 + e, 0, None))
    sqrt_1_minus_e = np.sqrt(np.clip(1 - e, 0, None))

    sin_E2 = np.sin(E / 2)
    cos_E2 = np.cos(E / 2)

    # Replace NaNs with 0 to avoid propagating errors
    sin_E2 = np.nan_to_num(sin_E2)
    cos_E2 = np.nan_to_num(cos_E2)

    return 2 * np.arctan2(
        sqrt_1_plus_e * sin_E2,
        sqrt_1_minus_e * cos_E2
    )

def radial_velocity(t, params, logP=True):
    """
    t : times at which to compute RV
    v0 : systemic velocity
    K : RV semiamplitude
    w : arg of periapsis
    e : eccentricity
    phi_0 : mean anomaly phase offset
    P : period (in same units as t)
    """
    v0, K, w, phi_0, e, P = params

    if logP==True:
        P=10**P

    M = 2 * np.pi * t / P - phi_0
    E = solve_kepler(M % (2 * np.pi), e)
    nu = true_anomaly(E, e)
    vr = v0 + K * (np.cos(nu + w) + e * np.cos(w))
    return vr

def generate_simulated_data(N):
    """
    returns RVs, obsEpochs, orbital parameters
    """
    orbital_params = generate_orbital_params(N)
    obstimes = obsEpochs(N)
    RVs_list = []
    for i in tqdm(range(N)):
        rvs = radial_velocity(obstimes[i], orbital_params[i], logP=True) + np.random.normal(0, 0.1, size=obstimes[i].shape)
        RVs_list.append(rvs)

    return np.array(RVs_list), obstimes, orbital_params


## function for phase folding
def compute_phase(t, P, T0=0.0):
    return ((t - T0) / P) % 1

