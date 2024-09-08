import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import get_gaia_period, get_star_photometry
from utils.period_utils import compute_period


class ObservationsOptimizer:
    def __init__(self, star_id: str, photometry_path: str, save_path: str, compute_combined: bool = False, compute_individuals: bool = True, load_from_file: bool = False):
        self.star_id = star_id
        self.photometry_path = photometry_path
        self.save_path = save_path
        self.compute_combined = compute_combined
        self.compute_individuals = compute_individuals
        self.load_from_file = load_from_file
        self.periods_by_observation_number = {}
        self.photometry_data = None

    def compute_period_for_observations(self, data: pd.DataFrame, observations: int) -> tuple:
        sample_data = data.sample(observations)
        return observations, compute_period(sample_data, 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')[0]

    def get_periods_by_observation_number(self, data: pd.DataFrame) -> dict:
        """
        Calculate the periods by number of observations for a given star,
        using a sample of the data with increasing size on each step.
        This method uses multithreading to speed up the process.
        """
        periods_by_observation_number = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            observation_sizes = list(range(5, len(data) + 5, 5))
            if observation_sizes[-1] <= len(data):
                observation_sizes.append(len(data))
            else:
                observation_sizes[-1] = len(data)

            futures = {executor.submit(
                self.compute_period_for_observations, data, observations): observations for observations in observation_sizes}

            for future in concurrent.futures.as_completed(futures):
                obs, period = future.result()
                periods_by_observation_number[obs] = period

        # Return sorted dict by observation number, since it may not be in order because of the multithreading
        return dict(sorted(periods_by_observation_number.items()))

    def find_min_observations(self, data: dict, std_threshold=0.005, window=5):
        """
        Find the minimum number of observations required to stabilize the period using rolling standard deviation.
        """
        observations = sorted(data.keys())
        periods = [data[o] for o in observations]

        period_series = pd.Series(periods)

        rolling_std = period_series.rolling(window=window).std()

        stable_idx = np.nonzero(rolling_std <= std_threshold)[0]

        if len(stable_idx) > 0:
            return observations[stable_idx[0]]
        else:
            return None

    def plot_periods_by_observation_number(self, title: str = None):
        """
        Plot the periods by observation number for a given star.
        """
        plt.figure().set_figwidth(10)
        plt.xlabel('Number of Observations')
        plt.ylabel('Period')
        if title is not None:
            plt.title(title)
        for key in self.periods_by_observation_number:
            print(f'Last period computed for {key}: {self.periods_by_observation_number[key][max(
                self.periods_by_observation_number[key].keys())]}')
            min_observations = self.find_min_observations(
                self.periods_by_observation_number[key])
            plt.plot(self.periods_by_observation_number[key].keys(),
                     self.periods_by_observation_number[key].values(), label=key)
            if min_observations is not None:
                print(f'Stabilization Point for {key}: {min_observations} with period {
                      self.periods_by_observation_number[key][min_observations]}')
                plt.plot([min_observations, min_observations], [0, 1],
                         label='Stabilization Point for ' + key, linestyle='--')
        gaia_period = get_gaia_period(self.star_id)
        print(f'Fetched Gaia Period: {gaia_period}')
        plt.plot([0, max(self.periods_by_observation_number[key].keys())], [gaia_period, gaia_period],
                 label='Gaia Period', linestyle='--')
        plt.legend()
        plt.show()

    def optimize_observations(self):
        """
        Optimize the number of observations needed to stabilize the period for the star.
        """
        if not self.load_from_file:
            self.photometry_data = get_star_photometry(
                self.photometry_path, self.star_id)
            combined_photometry = pd.concat(self.photometry_data.values())
            if self.compute_combined:
                self.periods_by_observation_number = {
                    "Combined": self.get_periods_by_observation_number(combined_photometry)
                }
            if self.compute_individuals:
                for key in self.photometry_data:
                    self.periods_by_observation_number[key] = self.get_periods_by_observation_number(
                        self.photometry_data[key])
            np.save(self.save_path, self.periods_by_observation_number)
        else:
            self.periods_by_observation_number = np.load(
                self.save_path + '.npy', allow_pickle=True).item()

        self.plot_periods_by_observation_number(
            self.star_id + ' Periods by Observation Number')

    def sequentially_add_observations_to_gaia(self):
        """
        Sequentially add our observations one by one to the Gaia data (combined)
        and plot it to see if the period stabilizes or takes erratic values at some point.
        """
        if not self.load_from_file:
            self.photometry_data = get_star_photometry(
                self.photometry_path, self.star_id)
            combined_gaia_photometry = pd.concat(
                [self.photometry_data[key] for key in self.photometry_data if key.startswith('Gaia-')])
            combined_own_photometry = pd.concat(
                [self.photometry_data[key] for key in self.photometry_data if not key.startswith('Gaia-')])

            self.periods_by_observation_number = {
                "Combined": {
                    len(combined_gaia_photometry): compute_period(combined_gaia_photometry, 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')[0]
                }
            }

            for n in range(10, len(combined_own_photometry) + 10, 10):
                if n > len(combined_own_photometry):
                    n = len(combined_own_photometry)
                print(f'Iteration {n} of {len(combined_own_photometry)}')
                combined_photometry = pd.concat(
                    [combined_gaia_photometry, combined_own_photometry.sample(n)])
                self.periods_by_observation_number["Combined"][len(combined_gaia_photometry) + n] = compute_period(
                    combined_photometry, 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO')[0]

            self.plot_periods_by_observation_number(
                'Sequentially adding our observations to Gaia\'s for ' + self.star_id)
            np.save(self.save_path + '_seq.npy',
                    self.periods_by_observation_number)
        else:
            self.periods_by_observation_number = np.load(
                self.save_path + '_seq.npy', allow_pickle=True).item()
            self.plot_periods_by_observation_number(
                'Sequentially adding our observations to Gaia\'s for ' + self.star_id)


if __name__ == '__main__':

    optimizer = ObservationsOptimizer(
        star_id='RR Gem',
        photometry_path='data/photometry/RR18_RRGem',
        save_path='data/periods/RRGem',
        compute_combined=True,
        compute_individuals=True,
        load_from_file=True
    )
    optimizer.optimize_observations()
    # optimizer.sequentially_add_observations_to_gaia()
