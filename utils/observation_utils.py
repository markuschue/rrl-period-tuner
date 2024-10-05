import astropy.units as u
import numpy as np


def get_exp_time(apparent_magnitude: float) -> int:
    if apparent_magnitude >= 16:
        return 60
    if apparent_magnitude >= 15:
        return 50
    if apparent_magnitude >= 14:
        return 20
    return 5


ZERO_POINT_FLUXES = {
    'U': 1823 * u.Jy,
    'B': 4130 * u.Jy,
    'V': 3781 * u.Jy,
    'R': 2941 * u.Jy,
    'I': 2635 * u.Jy,
    'J': 1603 * u.Jy,
    'H': 1075 * u.Jy,
    'K': 667 * u.Jy
}


def flux_to_magnitude(flux, band):
    """Convert flux to magnitude using the zero-point flux."""
    return -2.5 * np.log10(flux / ZERO_POINT_FLUXES[band])
