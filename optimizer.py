# Module used for optimizing the number of observations needed for a given star to reach a confident period estimation.
import os

import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table

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
    Get the periods by observation number for a given star.
    :param data: pd.DataFrame containing the data for the given star.
    :return: Dict containing the periods by observation number.
    """
    periods_by_observation_number = {}
    observations = 5
    while observations < len(data):
        periods_by_observation_number[observations], _ = compute_period(
            data[:observations], 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')
        observations += 5
    return periods_by_observation_number


def plot_periods_by_observation_number(periods_by_observation_number: pd.DataFrame):
    """
    Plot the periods by observation number for a given star.
    :param periods_by_observation_number: Dict containing the periods by observation number.
    """
    periods = [period for period in periods_by_observation_number.values()]
    observation_numbers = [
        observation_number for observation_number in periods_by_observation_number.keys()]
    plt.plot(observation_numbers, periods)
    plt.xlabel('Number of Observations')
    plt.ylabel('Period')
    plt.show()


if __name__ == '__main__':
    star_id = 'GAIA01_1321141977590168448'
    photometry_data = get_star_photometry(star_id)
    for key in photometry_data:
        periods_by_observation_number = get_periods_by_observation_number(
            photometry_data[key])
        plot_periods_by_observation_number(periods_by_observation_number)
