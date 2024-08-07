# Module used for optimizing the number of observations needed for a given star to reach a confident period estimation.
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_utils import get_star_lookup_data, get_star_photometry
from utils.period_utils import compute_period


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
    :param periods_by_observation_number: Dict containing the periods by observation number where each key is a filter, or combination of filters.
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
    STAR_ID = 'GAIA03_1360607637502749440'
    PHOTOMETRY_PATH = 'data/photometry/' + STAR_ID
    ONLY_COMBINED = False

    photometry_data = get_star_photometry(PHOTOMETRY_PATH)
    combined_photometry = pd.concat(photometry_data.values())
    periods_by_observation_number = {
        "Combined": get_periods_by_observation_number(combined_photometry)
    }
    if not ONLY_COMBINED:
        for key in photometry_data:
            periods_by_observation_number[key] = get_periods_by_observation_number(
                photometry_data[key])

    plot_periods_by_observation_number(
        periods_by_observation_number, get_gaia_period(STAR_ID), STAR_ID + ' Periods by Observation Number')
