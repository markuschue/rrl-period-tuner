import os
from pathlib import Path

import pandas as pd
from astropy.table import Table
from numpy import log

from utils.period_utils import prepareTable


def get_star_lookup_data(gaia_id: str) -> pd.DataFrame:
    """
    Get the star lookup data for a given Gaia ID.
    :param gaia_id: The Gaia ID to look up.
    :return: The star lookup data for the given Gaia ID.
    """
    star_lookup = pd.read_csv('data/star_lookup.csv')
    return star_lookup[star_lookup['source_id'] == int(gaia_id)]


def flux_to_magnitude_error(flux: float, flux_error: float) -> float:
    """
    Convert flux and flux error to magnitude and magnitude error.
    :param flux: The flux value.
    :param flux_error: The flux error value.
    :return: The magnitude error value.
    """
    return abs(-2.5 / flux / log(10)) * flux_error


def parse_gaia_photometry(gaia_photometry: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Gaia Epoch photometry data to a format that can be used by the optimizer.
    :param gaia_photometry: The Gaia Epoch photometry data.
    :return: The parsed Gaia photometry data.
    """
    gaia_photometry = gaia_photometry[~gaia_photometry['rejected_by_photometry']
                                      & ~gaia_photometry['rejected_by_variability']]
    gaia_photometry['MAG_AUTO_NORM'] = gaia_photometry['mag']
    gaia_photometry['MAGERR_AUTO'] = flux_to_magnitude_error(
        gaia_photometry['flux'], gaia_photometry['flux_error'])
    gaia_photometry['DATE-OBS'] = gaia_photometry['time']
    gaia_photometry['ID'] = gaia_photometry['source_id']
    return gaia_photometry[['ID', 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO']].copy()


def get_star_photometry(photometry_path: str) -> dict:
    """
    Get the prepared photometry data for a given star as a dict of dataframes for each filter.
    :param photometry_path: The path to the photometry data for the given star. 
    :return: The photometry data for the given star.
    """
    START_ID = Path(photometry_path).stem
    photometry_data = {}
    for photometry_file in os.listdir(photometry_path):
        if photometry_file.startswith('cat_'):
            photometry_data[photometry_file.split('.')[0].split(
                '_')[-1]] = prepareTable(photometry_path+'/'+photometry_file, 1)
        elif photometry_file.startswith('EPOCH_PHOTOMETRY-Gaia DR3'):
            photometry_data['gaia'] = parse_gaia_photometry(Table.read(
                photometry_path+'/'+photometry_file, hdu=1).to_pandas())
    varid = Table.read(photometry_path+"/finalIDs_" +
                       START_ID+".fits", hdu=1)[0][0]
    pd.options.mode.chained_assignment = None
    filtered_data = {}
    for key in photometry_data:
        if key != 'gaia':
            filtered_data[key] = photometry_data[key].loc[photometry_data[key].ID == varid, :]
        else:
            filtered_data[key] = photometry_data[key]
    return filtered_data
