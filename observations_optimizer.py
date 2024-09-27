import concurrent.futures
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import TimeDelta
from astropy.timeseries import LombScargle

from utils.data_utils import (combine_photometry_data, get_gaia_period,
                              get_mag_field_names, get_star_photometry,
                              parse_gaia_time)
from utils.period_utils import compute_period
from utils.run_utils import parse_cli_args


class ObservationsOptimizer:
    def __init__(self, star_id: str, photometry_path: str, save_path: str,
                 compute_combined: bool = False, compute_individuals: bool = True,
                 load_from_file: bool = False, observations_step: int = 5):
        self.star_id = star_id
        self.photometry_path = photometry_path
        self.save_path = save_path
        self.compute_combined = compute_combined
        self.compute_individuals = compute_individuals
        self.load_from_file = load_from_file
        self.periods_by_observation_number = {}
        self.observations_step = observations_step
        self.time_differences = [0, 5, 15, 30]
        # self.time_differences = [0]

        self.magstr, self.magerrstr = get_mag_field_names(
            photometry_path)
        print(f'Using magstr: {self.magstr}, magerrstr: {self.magerrstr}')
        self.datestr = 'DATE-OBS'

        self.data = get_star_photometry(
            self.photometry_path, self.star_id, magstr=self.magstr, magerrstr=self.magerrstr, datestr=self.datestr)
        own_data = {key: self.data[key]
                    for key in self.data if not key.startswith('Gaia')}
        gaia_data = {key: self.data[key]
                     for key in self.data if key.startswith('Gaia')}
        self.own_data = combine_photometry_data(own_data)
        self.gaia_data = combine_photometry_data(gaia_data)
        self.combined_data = combine_photometry_data(self.data)

    def filter_observations_by_time_difference(self, data: pd.DataFrame, time_difference: int) -> pd.DataFrame:
        """
        Filter the observations so that each observation has is separated by at least the given time difference.
        :param data: The photometry data for a given star, with the time difference column added.
        :param time_difference: The time difference between observations in minutes.
        """
        data = data.sort_values(by=self.datestr).reset_index(drop=True)
        time_difference = datetime.timedelta(minutes=time_difference)

        filtered_indices = []

        last_accepted_idx = 0
        filtered_indices.append(0)

        for current_idx in range(1, len(data)):
            current_time = parse_gaia_time(
                data.loc[current_idx, self.datestr])
            last_accepted_time = parse_gaia_time(
                data.loc[last_accepted_idx, self.datestr])

            if current_time - last_accepted_time >= TimeDelta(time_difference, format='datetime'):
                filtered_indices.append(current_idx)
                last_accepted_idx = current_idx

        return data.loc[filtered_indices].reset_index(drop=True)

    def compute_period_for_observations(self, data: pd.DataFrame, observations: int) -> tuple:
        """
        Compute the period for a sample of the data with the size of the given number of observations.
        :param data: The photometry data for a given star.
        :param observations: The number of observations to consider.
        """

        sample_data = data.sample(observations)
        best_period, self.freq, self.power = compute_period(
            sample_data, self.datestr, self.magstr, self.magerrstr)

        return observations, best_period

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
            if observation_sizes[-1] < len(data):
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

    def find_min_observations(self, data: dict, std_threshold=0.0000001, window=10):
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
        num_plots = len(self.time_differences)
        # Adjust the figure size based on the number of subplots
        fig, axs = plt.subplots(
            num_plots, 1, figsize=(10, 3 * num_plots))

        if title is not None:
            fig.suptitle(title)

        gaia_period = get_gaia_period(self.star_id)
        print(f'Fetched Gaia Period: {gaia_period}')

        for idx, time_diff in enumerate(self.time_differences):
            # Handle case when there's only one subplot
            ax = axs[idx] if num_plots > 1 else axs
            if idx == len(self.time_differences) - 1:
                ax.set_xlabel('Number of Observations')
            ax.set_ylabel('Period')
            ax.set_title(f'Minimum Time Difference: {time_diff}')

            for key in self.periods_by_observation_number[time_diff]:
                # Plot the periods by observation number for each key
                periods_data = self.periods_by_observation_number[time_diff][key]
                if len(periods_data) == 0:
                    continue
                ax.plot(periods_data.keys(), periods_data.values(), label=key)

                print(f'Last period computed for {key} at time difference {
                      time_diff}: {periods_data[max(periods_data.keys())]}')
                min_observations = self.find_min_observations(periods_data)

                if min_observations is not None:
                    print(f'Stabilization Point for {key} at time difference {time_diff}: {
                          min_observations} with period {periods_data[min_observations]}')
                    ax.plot([min_observations, min_observations], [min(periods_data.values()), max(
                        periods_data.values())], label=f'Stabilization Point for {key}', linestyle='--')

                ax.plot([0, max(periods_data.keys())], [gaia_period,
                        gaia_period], label='Gaia Period', linestyle='--')

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # Adjust the layout to prevent overlap with the title and legend
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.show()

    def optimize_observations(self):
        """
        Optimize the number of observations needed to stabilize the period for the star.
        """
        if not self.load_from_file:
            for time_diff in self.time_differences:
                if self.compute_combined:
                    self.periods_by_observation_number[time_diff] = {
                        "Combined": self.get_periods_by_observation_number(self.combined_data, time_diff)
                    }
                if self.compute_individuals:
                    for key in self.data:
                        if time_diff not in self.periods_by_observation_number:
                            self.periods_by_observation_number[time_diff] = {}
                        self.periods_by_observation_number[time_diff][key] = self.get_periods_by_observation_number(
                            self.data[key], time_diff)
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
            for time_diff in self.time_differences:
                self.periods_by_observation_number[time_diff] = {
                    "Combined": {
                        0: self.compute_period_for_observations(self.gaia_data, len(self.gaia_data))[1]
                    }
                }

                filtered_data = self.filter_observations_by_time_difference(
                    self.own_data, time_diff)
                for n in range(self.observations_step, len(filtered_data) + self.observations_step, self.observations_step):
                    if n > len(filtered_data):
                        n = len(filtered_data)
                    combined_photometry = pd.concat(
                        [self.gaia_data, filtered_data.sample(n)])
                    self.periods_by_observation_number[time_diff]["Combined"][n] = self.compute_period_for_observations(
                        combined_photometry, len(combined_photometry))[1]

            np.save(self.save_path + '_seq.npy',
                    self.periods_by_observation_number)
        else:
            self.periods_by_observation_number = np.load(
                self.save_path + '_seq.npy', allow_pickle=True).item()
        self.plot_periods_by_observation_number(
            'Sequentially adding our observations to Gaia\'s for ' + self.star_id)


if __name__ == '__main__':
    photometry_path, star_id = parse_cli_args(
        'Optimize the number of observations needed to stabilize the period for a given star')

    optimizer = ObservationsOptimizer(
        star_id=star_id,
        photometry_path=photometry_path,
        save_path='data/periods/latest',
        compute_combined=True,
        compute_individuals=False,
        load_from_file=False,
        observations_step=10
    )
    # optimizer.optimize_observations()
    optimizer.sequentially_add_observations_to_gaia()
