from datetime import datetime


class PhaseFinder():
    def __init__(self, period: float | None = None, reference_time: datetime | None = None):
        """
        Initialize the PhaseFinder with the period and reference time of the object
        :param period: The period of the object in days
        :param reference_time: The time where the object is at phase 0
        """
        self.period = period
        self.reference_time = reference_time

    def find_observation_phase(self, observation_time: datetime) -> float:
        """
        Find the phase of the object at the given observation time
        :param observation_time: The time of the observation
        """
        if self.period is None:
            raise ValueError("Period must be set before calling find_phase")
        if self.reference_time is None:
            raise ValueError(
                "Reference time must be set before calling find_phase")
        time_difference = observation_time - self.reference_time
        phase = time_difference.days / self.period % 1
        return phase


if __name__ == "__main__":
    period = 0.5
    reference_time = datetime(2017, 1, 1)
    phase_finder = PhaseFinder(period, reference_time)
    observation_time = datetime(2021, 1, 1)
    phase = phase_finder.find_observation_phase(observation_time)
    print(f"Phase: {phase}")
