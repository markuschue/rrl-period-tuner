import pandas as pd
from numpy import log


def get_star_lookup_data(gaia_id: str) -> pd.DataFrame:
    """
    Get the star lookup data for a given Gaia ID.
    :param gaia_id: The Gaia ID to look up.
    :return: The star lookup data for the given Gaia ID.
    """
    star_lookup = pd.read_csv('data/star_lookup.csv')
    return star_lookup[star_lookup['source_id'] == gaia_id]


def flux_to_magnitude_error(flux: float, flux_error: float) -> float:
    """
    Convert flux and flux error to magnitude and magnitude error.
    :param flux: The flux value.
    :param flux_error: The flux error value.
    :return: The magnitude error value.
    """
    return abs(-2.5 / flux / log(10)) * flux_error
