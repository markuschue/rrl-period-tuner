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
    Calculate the periods by number of observations for a given star, 
    using a sample of the data with increasing size on each step.
    :param data: pd.DataFrame containing the data for the given star.
    :return: Dict containing the periods by observation number.
    """
    periods_by_observation_number = {}
    observations = 5
    while observations < len(data):
        periods_by_observation_number[observations], _ = compute_period(
            data.sample(observations), 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')
        observations += 5
    return periods_by_observation_number


def plot_periods_by_observation_number(periods_by_observation_number: pd.DataFrame, title: str | None = None):
    """
    Plot the periods by observation number for a given star.
    :param periods_by_observation_number: Dict containing the periods by observation number.
    """
    plt.plot(periods_by_observation_number.keys(),
             periods_by_observation_number.values())
    plt.xlabel('Number of Observations')
    plt.ylabel('Period')
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    star_id = 'GAIA02_1204518424202454272'
    photometry_data = get_star_photometry(star_id)
    for key in photometry_data:
        periods_by_observation_number = get_periods_by_observation_number(
            photometry_data[key])
        plot_periods_by_observation_number(
            periods_by_observation_number, star_id + ' Periods by Observation Number for filter ' + key)
