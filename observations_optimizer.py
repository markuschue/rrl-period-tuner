import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

from utils.data_utils import get_gaia_period, get_star_photometry
from utils.period_utils import compute_period


class ObservationsOptimizer:
    def __init__(self, star_id: str, photometry_path: str, save_path: str, compute_combined: bool = False, compute_individuals: bool = True, load_from_file: bool = False, observations_step: int = 5):
        self.star_id = star_id
        self.photometry_path = photometry_path
        self.save_path = save_path
        self.compute_combined = compute_combined
        self.compute_individuals = compute_individuals
        self.load_from_file = load_from_file
        self.periods_by_observation_number = {}
        self.photometry_data = None
        self.observations_step = observations_step
        self.time_differences = [5, 10, 15, 20, 30, 45, 60]

    def add_time_difference_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add a column to the data with the time difference between observations.
        :param data: The photometry data for a given star.
        """
        data['DATETIME'] = Time(data['DATE-OBS'], format='mjd').to_datetime()
        data = data.sort_values('DATETIME')
        data['TIME_DIFF'] = data['DATETIME'].diff()
        return data

    def filter_observations_by_time_difference(self, data: pd.DataFrame, time_difference: int) -> pd.DataFrame:
        """
        Filter the observations by a given time difference between them.
        :param data: The photometry data for a given star, with the time difference column added.
        :param time_difference: The time difference between observations in minutes.
        """
        time_diff_delta = pd.Timedelta(minutes=time_difference)
        return data[data['TIME_DIFF'] >= time_diff_delta]

    def compute_period_for_observations(self, data: pd.DataFrame, observations: int) -> tuple:
        """
        Compute the period for a sample of the data with the size of the given number of observations.
        :param data: The photometry data for a given star.
        :param observations: The number of observations to consider.
        """

        sample_data = data.sample(observations)

        return observations, compute_period(sample_data, 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO', False)[0]

    def get_periods_by_observation_number(self, data: pd.DataFrame, time_difference: int) -> dict:
        """
        Calculate the periods by number of observations for a given star,
        using a sample of the data with increasing size on each step.
        This method uses multithreading to speed up the process.
        """
        periods_by_observation_number = {}
        data = self.filter_observations_by_time_difference(
            data, time_difference)
        print(f'Computing periods for {len(data)} observations\n0/{len(data)}')
        if len(data) < 5:
            return periods_by_observation_number

        with concurrent.futures.ThreadPoolExecutor() as executor:
            observation_sizes = list(range(self.observations_step, len(
                data) + self.observations_step, self.observations_step))
            if observation_sizes[-1] <= len(data):
                observation_sizes.append(len(data))
            else:
                observation_sizes[-1] = len(data)

            futures = {executor.submit(
                self.compute_period_for_observations, data, observations): observations for observations in observation_sizes}

            for future in concurrent.futures.as_completed(futures):
                obs, period = future.result()
                periods_by_observation_number[obs] = period
                print(f'{obs}/{len(data)}')

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
        Plot the periods by observation number for each time difference in self.time_differences.
        """
        num_time_diffs = len(self.time_differences)
        # Adjust the figure size based on the number of subplots
        fig, axs = plt.subplots(
            num_time_diffs, figsize=(10, 5 * num_time_diffs))

        if title is not None:
            fig.suptitle(title)

        for idx, time_diff in enumerate(self.time_differences):
            # Handle case when there's only one subplot
            ax = axs[idx] if num_time_diffs > 1 else axs
            ax.set_xlabel('Number of Observations')
            ax.set_ylabel('Period')
            ax.set_title(f'Time Difference: {time_diff}')

            for key in self.periods_by_observation_number[time_diff]:
                # Plot the periods by observation number for each key
                periods_data = self.periods_by_observation_number[time_diff][key]
                ax.plot(periods_data.keys(), periods_data.values(), label=key)

                print(f'Last period computed for {key} at time difference {
                      time_diff}: {periods_data[max(periods_data.keys())]}')
                min_observations = self.find_min_observations(periods_data)

                if min_observations is not None:
                    print(f'Stabilization Point for {key} at time difference {time_diff}: {
                          min_observations} with period {periods_data[min_observations]}')
                    ax.plot([min_observations, min_observations], [
                            0, 1], label=f'Stabilization Point for {key}', linestyle='--')

            # Plot Gaia period for reference
            gaia_period = get_gaia_period(self.star_id)
            print(f'Fetched Gaia Period: {gaia_period}')
            ax.plot([0, max(periods_data.keys())], [gaia_period,
                    gaia_period], label='Gaia Period', linestyle='--')

            ax.legend()

        # Adjust the layout to prevent overlap with the title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def optimize_observations(self):
        """
        Optimize the number of observations needed to stabilize the period for the star.
        """
        if not self.load_from_file:
            for time_diff in self.time_differences:
                self.photometry_data = get_star_photometry(
                    self.photometry_path, self.star_id)
                if self.compute_combined:
                    combined_photometry = self.add_time_difference_column(
                        pd.concat(self.photometry_data.values()))
                    self.periods_by_observation_number[time_diff] = {
                        "Combined": self.get_periods_by_observation_number(combined_photometry, time_diff)
                    }
                if self.compute_individuals:
                    for key in self.photometry_data:
                        self.periods_by_observation_number[key] = self.get_periods_by_observation_number(
                            self.add_time_difference_column(self.photometry_data[key]), time_diff)
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

            for n in range(self.observations_step, len(combined_own_photometry) + self.observations_step, self.observations_step):
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
        compute_combined=False,
        compute_individuals=True,
        load_from_file=False,
        observations_step=50
    )
    optimizer.optimize_observations()
    # optimizer.sequentially_add_observations_to_gaia()
