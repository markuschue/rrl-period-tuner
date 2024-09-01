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


def bjd_tcb_to_mjd(bjd_tcb_date: float) -> float:
    # Reference JD corresponding to T0 (2010-01-01T00:00:00)
    ref_jd = 2455197.5

    # Calculate the full BJD(TCB)
    bjd_tcb = bjd_tcb_date + ref_jd

    # Convert BJD(TCB) to MJD
    mjd = bjd_tcb - 2400000.5

    return mjd


def parse_gaia_photometry(gaia_photometry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Parse the Gaia Epoch photometry data to a format that can be used by the optimizer.
    :param gaia_photometry: The Gaia Epoch photometry data.
    :return: The parsed Gaia photometry data in a dictionary with the filter as the key.
    """
    gaia_photometry = gaia_photometry[~gaia_photometry['rejected_by_photometry']
                                      & ~gaia_photometry['rejected_by_variability']]
    gaia_photometry['MAG_AUTO_NORM'] = gaia_photometry['mag']
    gaia_photometry['MAGERR_AUTO'] = flux_to_magnitude_error(
        gaia_photometry['flux'], gaia_photometry['flux_error'])
    gaia_photometry['DATE-OBS'] = bjd_tcb_to_mjd(gaia_photometry['time'])
    gaia_photometry['ID'] = gaia_photometry['source_id']
    gaia_photometry_data = {}
    for filter_key in gaia_photometry['band'].unique():
        parsed_filter_key = str(filter_key)
        gaia_photometry_data[parsed_filter_key] = gaia_photometry[gaia_photometry['band'] == filter_key]
        gaia_photometry_data[parsed_filter_key] = gaia_photometry_data[parsed_filter_key][[
            'ID', 'DATE-OBS', 'MAG_AUTO_NORM', 'MAGERR_AUTO']].copy()
    return gaia_photometry_data


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
            gaia_photometry = parse_gaia_photometry(Table.read(
                photometry_path+'/'+photometry_file, hdu=1).to_pandas())
            for filter_key in gaia_photometry:
                photometry_data['Gaia-' +
                                filter_key] = gaia_photometry[filter_key]
    varid = Table.read(photometry_path+"/finalIDs_" +
                       START_ID+".fits", hdu=1)[0][0]
    pd.options.mode.chained_assignment = None
    filtered_data = {}
    for key in photometry_data:
        if 'Gaia' not in key:
            filtered_data[key] = photometry_data[key].loc[photometry_data[key].ID == varid, :]
        else:
            filtered_data[key] = photometry_data[key]
    return filtered_data
