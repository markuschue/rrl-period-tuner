import multiprocessing as mp
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.timeseries import LombScargle, LombScargleMultiband


def compute_period(data, datestr, magstr, magerrstr):
    epoch, mag, mag_err = [data[datestr].values,
                           data[magstr].values, data[magerrstr].values]
    NN = 2000
    period_array = np.zeros(NN)

    print("")
    print("Performing bootstrap to compute best period...")
    for ii in range(NN):
        rng = np.random.default_rng(seed=42)
        loc_new = rng.choice(len(mag), replace=False, size=int(len(mag)*0.4))
        epoch_new = epoch[loc_new]
        mag_new = mag[loc_new]
        mag_err_new = mag_err[loc_new]

        freq, power = LombScargle(epoch_new, mag_new, mag_err_new, fit_mean=True).autopower(
            minimum_frequency=1, maximum_frequency=5)
        # power = LombScargle(epoch_new, mag_new, mag_err_new).power(freq,assume_regular_frequency=True)
        best_freq = freq[np.argmax(power)]
        period_array[ii] = 1/best_freq

        if (ii % 50) == 0:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            perc = int(np.round(ii/NN * 100))
            sys.stdout.write("[%-20s] %d%%" % ('='*perc, perc))
            sys.stdout.flush()

    possible_periods, counts = np.unique(
        np.sort(np.round(period_array, 5)), return_counts=True)
    possible_periods = np.c_[possible_periods, counts]

    best_period = sp.stats.mode(np.round(period_array, 9))[0]
    sys.stdout.write('\r')
    return best_period, np.flip(possible_periods[np.argsort(possible_periods[:, 1])], axis=0)[:8]


def initialize_worker(seed):
    np.random.seed(seed)


def compute_lombscargle_period(seed, epoch, mag, mag_err):
    np.random.seed(seed)
    loc_new = np.random.choice(
        len(mag), size=int(len(mag) * 0.4), replace=False)
    epoch_new = epoch[loc_new]
    mag_new = mag[loc_new]
    mag_err_new = mag_err[loc_new]

    freq, power = LombScargle(epoch_new, mag_new, mag_err_new,  fit_mean=True).autopower(
        minimum_frequency=1.1, maximum_frequency=4)
    best_freq = freq[np.argmax(power)]
    period = 1 / best_freq
    return period


def compute_period_nb(data, datestr, magstr, magerrstr, seed=42):
    epoch, mag, mag_err = [data[datestr].values,
                           data[magstr].values, data[magerrstr].values]

    NN = 2000
    print("")
    print("Performing bootstrap to compute best period...")
    # Create seeds for each process to ensure they are different
    seeds = np.random.RandomState(seed).randint(0, 2**32 - 1, size=NN)

    with mp.Pool(mp.cpu_count(), initializer=initialize_worker, initargs=(seed,)) as pool:
        results = pool.starmap(compute_lombscargle_period, [
                               (s, epoch, mag, mag_err) for s in seeds])

    period_array = np.array(results)

    possible_periods, counts = np.unique(
        np.sort(np.round(period_array, 5)), return_counts=True)
    possible_periods = np.c_[possible_periods, counts]

    best_period = sp.stats.mode(np.round(period_array, 9))[0]
    return best_period, np.flip(possible_periods[np.argsort(possible_periods[:, 1])], axis=0)[:8]
