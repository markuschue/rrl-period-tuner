# Module used for optimizing the number of observations needed for a given star to reach a confident period estimation.
import os

import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table

from utils.data_utils import get_star_lookup_data
from utils.period_utils import compute_period, prepareTable


def get_star_photometry(star_id: str) -> list[pd.DataFrame]:
    """
    Get the prepared photometry data for a given star.
    :param star_id: The ID of the star to get the photometry data for.
    :return: The photometry data for the given star.
    """
    PHOTOMETRY_PATH = './data/photometry/' + star_id
    photometry_data = {}
    for photometry_file in os.listdir(PHOTOMETRY_PATH):
        if photometry_file.startswith('cat_'):
            photometry_data[photometry_file.split('.')[0].split(
                '_')[-1]] = prepareTable(PHOTOMETRY_PATH+'/'+photometry_file, 1)
    varid = Table.read(PHOTOMETRY_PATH+"/finalIDs_" +
                       star_id+".fits", hdu=1)[0][0]
    filtered_data = {}
    pd.options.mode.chained_assignment = None
    for key in photometry_data:
        filtered_data[key] = photometry_data[key].loc[photometry_data[key].ID == varid, :]
    return filtered_data


def get_periods_by_observation_number(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the periods by number of observations for a given star,
    using a sample of the data with increasing size on each step.
    :param data: pd.DataFrame containing the data for the given star.
    :return: Dict containing the periods by observation number.
    """
    periods_by_observation_number = {}
    observations = 5
    while observations < len(data) + 5:
        if observations > len(data):
            observations = len(data)
        periods_by_observation_number[observations], _ = compute_period(
            data.sample(observations), 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')
        observations += 5
    return periods_by_observation_number


def get_gaia_period(star_id: str) -> float:
    """
    Get the period of a star using Gaia data.
    :param star_id: The ID of the star to get the period for.
    :return: The period of the star.
    """
    gaia_id = star_id.split('_')[-1]
    gaia_data = get_star_lookup_data(gaia_id)
    return gaia_data['rrl.pf'].values[0]


def plot_periods_by_observation_number(periods_by_observation_number: dict, gaia_period: float, title: str = None):
    """
    Plot the periods by observation number for a given star.
    :param periods_by_observation_number: Dict containing the periods by observation number where each key is a filter, and its value a pd.DataFrame.
    :param title: The title of the plot.
    """
    plt.figure().set_figwidth(10)
    plt.xlabel('Number of Observations')
    plt.ylabel('Period')
    if title is not None:
        plt.title(title)
    for key in periods_by_observation_number:
        plt.plot(periods_by_observation_number[key].keys(),
                 periods_by_observation_number[key].values(), label=key)
    plt.plot([0, max(periods_by_observation_number[key].keys())], [gaia_period, gaia_period],
             label='Gaia Period', linestyle='--')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    star_id = 'GAIA03_1360607637502749440'
    photometry_data = get_star_photometry(star_id)
    periods_by_observation_number = {}
    for key in photometry_data:
        periods_by_observation_number[key] = get_periods_by_observation_number(
            photometry_data[key])
    plot_periods_by_observation_number(
        periods_by_observation_number, get_gaia_period(star_id), star_id + ' Periods by Observation Number')
