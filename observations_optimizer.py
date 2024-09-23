import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

from utils.data_utils import (combine_photometry_data, get_gaia_period,
                              get_mag_field_names, get_star_photometry,
                              parse_gaia_time)
from utils.period_utils import compute_period


class ObservationsOptimizer:
    def __init__(self, star_id: str, photometry_path: str, save_path: str,
                 compute_combined: bool = False, compute_individuals: bool = True,
                 load_from_file: bool = False, observations_step: int = 5,
                 magstr: str = 'MAG_AUTO_NORM', magerrstr: str = 'MAGERR_AUTO', datestr: str = "DATE-OBS"):
        self.star_id = star_id
        self.photometry_path = photometry_path
        self.save_path = save_path
        self.compute_combined = compute_combined
        self.compute_individuals = compute_individuals
        self.load_from_file = load_from_file
        self.periods_by_observation_number = {}
        self.photometry_data = None
        self.observations_step = observations_step
        self.time_differences = [0, 30, 60, 120, 1440]
        self.magstr = magstr
        self.magerrstr = magerrstr
        self.datestr = datestr

    def filter_observations_by_time_difference(self, data: pd.DataFrame, time_difference: int) -> pd.DataFrame:
        """
        Filter the observations so that each observation has is separated by at least the given time difference.
        :param data: The photometry data for a given star, with the time difference column added.
        :param time_difference: The time difference between observations in minutes.
        """
        time_diff_delta = pd.Timedelta(minutes=time_difference)
        filtered_data = data.copy()
        for idx, row in data.iterrows():
            if idx == 0:
                continue
            if idx - 1 in data.index and parse_gaia_time(row[self.datestr]) - parse_gaia_time(data.loc[idx - 1, self.datestr]) < time_diff_delta:
                filtered_data.drop(idx, inplace=True)
        return filtered_data

    def compute_period_for_observations(self, data: pd.DataFrame, observations: int) -> tuple:
        """
        Compute the period for a sample of the data with the size of the given number of observations.
        :param data: The photometry data for a given star.
        :param observations: The number of observations to consider.
        """

        sample_data = data.sample(observations)
        self.freq, self.power = LombScargle(sample_data[self.datestr], sample_data[self.magstr], sample_data[self.magerrstr]).autopower(
            minimum_frequency=1, maximum_frequency=5)

        best_period = 1/self.freq[np.argmax(self.power)]

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
        num_plots = len(self.time_differences)
        # Adjust the figure size based on the number of subplots
        fig, axs = plt.subplots(
            num_plots, figsize=(10, 5 * num_plots))

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
                self.photometry_data = get_star_photometry(
                    self.photometry_path, self.star_id, magstr=self.magstr, magerrstr=self.magerrstr, datestr=self.datestr)
                if self.compute_combined:
                    combined_photometry = combine_photometry_data(
                        self.photometry_data)
                    self.periods_by_observation_number[time_diff] = {
                        "Combined": self.get_periods_by_observation_number(combined_photometry, time_diff)
                    }
                if self.compute_individuals:
                    for key in self.photometry_data:
                        if time_diff not in self.periods_by_observation_number:
                            self.periods_by_observation_number[time_diff] = {}
                        self.periods_by_observation_number[time_diff][key] = self.get_periods_by_observation_number(
                            self.photometry_data[key], time_diff)
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
                    len(combined_gaia_photometry): compute_period(combined_gaia_photometry, self.datestr, self.magstr, self.magerrstr)[0]
                }
            }

            for n in range(self.observations_step, len(combined_own_photometry) + self.observations_step, self.observations_step):
                if n > len(combined_own_photometry):
                    n = len(combined_own_photometry)
                print(f'Iteration {n} of {len(combined_own_photometry)}')
                combined_photometry = pd.concat(
                    [combined_gaia_photometry, combined_own_photometry.sample(n)])
                self.periods_by_observation_number["Combined"][len(combined_gaia_photometry) + n] = compute_period(
                    combined_photometry, self.datestr, self.magstr, self.magerrstr)[0]

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
    if len(os.sys.argv) < 2:
        print("Usage: python observations_optimizer.py <path_to_photometry> <optional: star name>")
        os.sys.exit(1)
    elif not os.path.exists(os.sys.argv[1]):
        print("Path does not exist")
        os.sys.exit(1)
    elif len(os.sys.argv) == 3:
        photometry_path: str = os.sys.argv[1]
        star_id: str = os.sys.argv[2]
    else:
        photometry_path: str = os.sys.argv[1]
        if photometry_path.endswith("/"):
            photometry_path = photometry_path[:-1]
        star_id = photometry_path.split("/")[-1]
        if '_' in star_id:
            star_id = star_id.split("_")[1]
        if 'gaia' in photometry_path.lower():
            star_id = 'Gaia DR3 ' + star_id
    magstr, magerrstr = get_mag_field_names(
        photometry_path)
    print(f'Using magstr: {magstr}, magerrstr: {magerrstr}')

    optimizer = ObservationsOptimizer(
        star_id=star_id,
        photometry_path=photometry_path,
        save_path='data/periods/latest',
        compute_combined=True,
        compute_individuals=False,
        load_from_file=False,
        observations_step=10,
        magstr=magstr,
        magerrstr=magerrstr
    )
    optimizer.optimize_observations()
    # optimizer.sequentially_add_observations_to_gaia()
