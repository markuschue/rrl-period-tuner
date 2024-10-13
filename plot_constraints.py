from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroplan import (AirmassConstraint, AltitudeConstraint,
                       AtNightConstraint, FixedTarget,
                       MoonSeparationConstraint)
from astroplan.scheduling import Scheduler
from astroplan.utils import time_grid_from_range
from astropy.time import Time

from observations_planner import ObservationPlanner

with open('data/scheduler_a.pkl', 'rb') as file:
    planner: ObservationPlanner = pickle.load(file)

    observer = planner.observer

    target = FixedTarget.from_name("V* V732 Her")

    constraints = [AltitudeConstraint(30*u.deg, 90*u.deg),
                   AtNightConstraint.twilight_astronomical(),
                   AirmassConstraint(max=2.5),
                   MoonSeparationConstraint(min=15*u.deg)]

    # Define range of times to observe between
    start_time = planner.time_start
    end_time = planner.time_end
    time_resolution = 1 * u.hour

    # Create grid of times from ``start_time`` to ``end_time``
    # with resolution ``time_resolution``
    time_grid = time_grid_from_range([start_time, end_time],
                                     time_resolution=time_resolution)

    observability_grid = np.zeros((len(constraints), len(time_grid)))

    for i, constraint in enumerate(constraints):
        # Evaluate each constraint
        observability_grid[i, :] = constraint(
            observer, target, times=time_grid)

    # Create plot showing observability of the target:

    n_constraints = len(constraints)
    extent = [-0.5, -0.5 + len(time_grid), -0.5, n_constraints - 0.5]

    # Create plot showing observability of the target
    # Adjust the height dynamically
    fig, ax = plt.subplots(figsize=(10, n_constraints * 1.5))
    ax.imshow(observability_grid, extent=extent, aspect='auto')

    # Dynamically set y-ticks based on the number of constraints
    ax.set_yticks(range(n_constraints))
    ax.set_yticklabels([c.__class__.__name__ for c in reversed(constraints)])

    # Set x-ticks and labels for the time grid
    ax.set_xticks(range(len(time_grid)))
    ax.set_xticklabels([t.datetime.strftime("%H:%M") for t in time_grid])

    # Minor ticks for better grid visibility
    ax.set_xticks(np.arange(extent[0], extent[1]), minor=True)
    ax.set_yticks(np.arange(extent[2], extent[3]), minor=True)

    # Grid configuration
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(axis='x', which='minor', bottom=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.tick_params(axis='y', which='minor', left=False)

    # Set labels and adjust layout
    ax.set_xlabel('Time on {0} UTC'.format(time_grid[0].datetime.date()))
    fig.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.2)

    plt.show()
