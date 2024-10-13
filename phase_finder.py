from datetime import datetime

from astropy.time import Time

from utils.data_utils import (combine_photometry_data, get_gaia_period,
                              get_mag_field_names, get_star_photometry,
                              parse_gaia_time)
from utils.period_utils import compute_period
from utils.run_utils import parse_cli_args


class PhaseFinder():
    def __init__(self, star_id: str, photometry_path: str):
        self.star_id = star_id
        self.photometry_path = photometry_path

        self.magstr, self.magerrstr = get_mag_field_names(
            photometry_path)
        print(f'Using magstr: {self.magstr}, magerrstr: {self.magerrstr}')
        self.datestr = 'DATE-OBS'

        self.photometry_data = get_star_photometry(
            photometry_path=photometry_path, star_id=star_id,
            magstr=self.magstr, magerrstr=self.magerrstr, datestr=self.datestr)

        self.combined_data = combine_photometry_data(self.photometry_data)

        self.period = compute_period(
            self.combined_data, self.datestr, self.magstr, self.magerrstr, get_gaia_period(self.star_id))[0]

        # Reference time will be the time where the magnitude is maximum
        self.reference_time = parse_gaia_time(
            self.combined_data.iloc[self.combined_data[self.magstr].idxmax(
            )][self.datestr]
        ).to_datetime()

    def days_to_seconds(self, days: float):
        return days * 24 * 3600

    def find_observation_phase(self, observation_time: datetime) -> float:
        """
        Find the phase of the object at the given observation time
        :param observation_time: The time of the observation
        """
        time_difference = observation_time - self.reference_time
        phase = (time_difference.total_seconds() /
                 self.days_to_seconds(self.period)) % 1
        return phase


if __name__ == "__main__":
    photometry_path, star_id = parse_cli_args(
        'Find the phase of an observation for a given star')
    observation_time = datetime.strptime(
        '2024.8.06 9:32:06', '%Y.%m.%d %H:%M:%S')
    phase_finder = PhaseFinder(star_id, photometry_path)
    print(f'Found period: {phase_finder.period}')
    print(f'Fase of the star at the time of {observation_time.strftime(
        "%d/%m/%Y, %H:%M:%S")}: {phase_finder.find_observation_phase(observation_time)}')
