import pickle

import astropy.units as u
import numpy as np
import pandas as pd
from astroplan import (AirmassConstraint, AltitudeConstraint,
                       AtNightConstraint, FixedTarget,
                       MoonSeparationConstraint, Observer, ObservingBlock,
                       Schedule, SequentialScheduler, Transitioner)
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta
from astroquery.simbad import Simbad

from utils.data_utils import get_gaia_period
from utils.observation_utils import flux_to_magnitude, get_exp_time

# In seconds
MAX_TIME_PER_STAR = 30 * u.minute
READOUT_TIME = 20 * u.second


class ObservationPlanner:
    def __init__(self, star_ids, date, location):
        """
        Initialize with star IDs, observation date, and Earth location (longitude, latitude, altitude).
        """
        self.star_ids = star_ids
        self.date = Time(date)
        self.location = EarthLocation(
            lat=location['latitude'], lon=location['longitude'], height=location['altitude'])
        self.observer = Observer(
            location=self.location, name="Observer", timezone="UTC")
        self.star_data = []
        self.blocks = []
        self.schedule = None

        self.time_start = self.observer.twilight_evening_astronomical(
            self.date)  # Evening
        self.time_end = self.observer.twilight_morning_astronomical(
            self.date)  # Morning

        self.constraints = [AltitudeConstraint(30*u.deg, 90*u.deg),
                            AtNightConstraint.twilight_astronomical(),
                            AirmassConstraint(max=2.5),
                            MoonSeparationConstraint(min=15*u.deg)]

    def fetch_star_data(self):
        """
        Fetches star data (coordinates, magnitude, etc.) using astroquery (Simbad, etc.).
        """
        simbad = Simbad()
        simbad.add_votable_fields('coordinates', 'flux(V)')
        result = simbad.query_objects(self.star_ids)
        for star in result:
            name = star['MAIN_ID']
            ra = star['RA']
            dec = star['DEC']
            mag = star['FLUX_V']
            self.star_data.append({
                'name': name,
                'coord': SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg)),
                'magnitude': mag,
                'period': get_gaia_period(name)
            })

    def create_observing_blocks(self):
        # Create observing blocks for each star based on their magnitude
        observed_time: Time = self.time_start

        while observed_time < self.time_end:
            for star in self.star_data:
                target = FixedTarget(coord=star['coord'], name=star['name'])
                exp_time = get_exp_time(star['magnitude']) * u.second
                single_obs_time = exp_time + READOUT_TIME
                n_obs = int(MAX_TIME_PER_STAR.to(u.second) // single_obs_time)

                for i in range(n_obs):
                    block = ObservingBlock(
                        target, single_obs_time, configuration=None, priority=i + int(star['period'] * 100))
                    self.blocks.append(block)
                    observed_time += TimeDelta(single_obs_time)

    def plan_schedule(self):
        # Not needed if we observe only one filter
        transitioner = Transitioner(slew_rate=1*u.deg / u.second)

        scheduler = SequentialScheduler(
            constraints=self.constraints, observer=self.observer, transitioner=transitioner)
        self.schedule = Schedule(self.time_start, self.time_end)

        # Takes a long time, but it's normal
        scheduler(self.blocks, self.schedule)
        with open('data/scheduler.pkl', 'wb') as file:
            pickle.dump(self, file)

    def get_observation_table(self):
        table = []
        for block in self.schedule.scheduled_blocks:
            if isinstance(block, ObservingBlock):
                table.append({
                    'star': block.target.name,
                    'start_time': block.start_time.iso,
                    'end_time': block.end_time.iso,
                    'duration': block.duration.to(u.second).value
                })
        return pd.DataFrame(table)

    def run(self):
        self.fetch_star_data()
        self.create_observing_blocks()
        self.plan_schedule()
        return self.get_observation_table()


if __name__ == '__main__':
    star_ids = ['U Lep', 'RR Gem']
    date = '2024-10-03'
    location = {'latitude': 28.2996667,
                'longitude': -16.511027777777777,
                'altitude': 2381.25}  # IAC80 Location

    planner = ObservationPlanner(star_ids, date, location)
    observation_table = planner.run()
    observation_table.to_csv('data/observation_table.csv')
