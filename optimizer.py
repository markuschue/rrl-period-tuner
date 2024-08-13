# Module used for optimizing the number of observations needed for a given star to reach a confident period estimation.
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import get_star_lookup_data, get_star_photometry
from utils.period_utils import compute_period


def compute_period_for_observations(data: pd.DataFrame, observations: int) -> tuple:
    sample_data = data.sample(observations)
    return observations, compute_period(sample_data, 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')[0]


def get_periods_by_observation_number(data: pd.DataFrame) -> dict:
    """
    Calculate the periods by number of observations for a given star,
    using a sample of the data with increasing size on each step.
    :param data: pd.DataFrame containing the data for the given star.
    :return: Dict containing the periods by observation number where 
        the key is the number of observations and the value is the 
        period obtained.
    """
    periods_by_observation_number = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare observation sizes
        observation_sizes = list(range(5, len(data) + 5, 5))
        if observation_sizes[-1] <= len(data):
            observation_sizes.append(len(data))
        else:
            observation_sizes[-1] = len(data)

        # Map each observation size to a future task
        futures = {executor.submit(
            compute_period_for_observations, data, observations): observations for observations in observation_sizes}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            obs, period = future.result()
            periods_by_observation_number[obs] = period

    # Return sorted dict by observation number, since it may not be in order because of the multithreading
    return dict(sorted(periods_by_observation_number.items()))


def get_gaia_period(star_id: str) -> float:
    """
    Get the period of a star using Gaia data.
    :param star_id: The ID of the star to get the period for.
    :return: The period of the star.
    """
    gaia_id = star_id.split('_')[-1]
    gaia_data = get_star_lookup_data(gaia_id)
    return gaia_data['rrl.pf'].values[0]


def find_min_observations(data: dict, std_threshold=0.005, window=5):
    """
    Find the minimum number of observations required to stabilize the period using rolling standard deviation.

    Parameters:
        data (dict): A dictionary where keys are the number of observations, and values are the periods.
        std_threshold (float): The maximum allowed rolling standard deviation for stabilization.
        window (int): The number of consecutive observations used in the rolling window.

    Returns:
        int: The minimum number of observations required to stabilize the period.
    """
    observations = sorted(data.keys())
    periods = [data[o] for o in observations]

    # Convert periods to a Pandas Series for rolling operations
    period_series = pd.Series(periods)

    # Calculate the rolling standard deviation
    rolling_std = period_series.rolling(window=window).std()

    # Find the first point where the rolling standard deviation is below the threshold
    stable_idx = np.nonzero(rolling_std <= std_threshold)[0]

    if len(stable_idx) > 0:
        return observations[stable_idx[0]]
    else:
        return None  # Return None if no stabilization is found


def plot_periods_by_observation_number(periods_by_observation_number: dict, gaia_period: float, title: str = None):
    """
    Plot the periods by observation number for a given star.
    :param periods_by_observation_number: Dict containing the periods by observation number where each key is a filter, or combination of filters.
    :param title: The title of the plot.
    """
    plt.figure().set_figwidth(10)
    plt.xlabel('Number of Observations')
    plt.ylabel('Period')
    if title is not None:
        plt.title(title)
    for key in periods_by_observation_number:
        print(f'Last period found for {key}: {periods_by_observation_number[key][max(
            periods_by_observation_number[key].keys())]}')
        min_observations = find_min_observations(
            periods_by_observation_number[key])
        plt.plot(periods_by_observation_number[key].keys(),
                 periods_by_observation_number[key].values(), label=key)
        if min_observations is not None:
            print(f'Stabilization Point for {key}: {min_observations} with period {
                  periods_by_observation_number[key][min_observations]}')
            plt.plot([min_observations, min_observations], [0, 1],
                     label='Stabilization Point for ' + key, linestyle='--')
    plt.plot([0, max(periods_by_observation_number[key].keys())], [gaia_period, gaia_period],
             label='Gaia Period', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    STAR_ID = 'GAIA03_1360607637502749440'
    PHOTOMETRY_PATH = 'data/photometry/' + STAR_ID
    PERIODS_PATH = 'data/periods/' + STAR_ID
    ONLY_COMBINED = False
    LOAD_FROM_FILE = True

    periods_by_observation_number = {}

    if not LOAD_FROM_FILE:
        photometry_data = get_star_photometry(PHOTOMETRY_PATH)
        # combined_photometry = pd.concat(photometry_data.values())
        # periods_by_observation_number = {
        #     "Combined": get_periods_by_observation_number(combined_photometry)
        # }
        if not ONLY_COMBINED:
            for key in photometry_data:
                periods_by_observation_number[key] = get_periods_by_observation_number(
                    photometry_data[key])
        np.save(PERIODS_PATH, periods_by_observation_number)
    else:
        periods_by_observation_number = np.load(
            PERIODS_PATH + '.npy', allow_pickle=True).item()

    plot_periods_by_observation_number(
        periods_by_observation_number, get_gaia_period(STAR_ID), STAR_ID + ' Periods by Observation Number')
