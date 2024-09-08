import os
from pathlib import Path

import pandas as pd
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
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


def get_star_gaia_id(star_name: str) -> str:
    """
    Transform a star name or identifier to a Gaia DR3 ID
    using the Simbad service.
    :param star_name: The name or identifier of the star.
    """
    Simbad.add_votable_fields('ids')
    result_table = Simbad.query_object(star_name)
    if result_table is not None:
        gaia_id = None
        for row in result_table['IDS'][0].split('|'):
            if 'Gaia DR3' in row:
                gaia_id = row.split()[-1]
                break
        if gaia_id is not None:
            return gaia_id
    raise ValueError(f'No star found with the name {star_name}')


def parse_gaia_photometry(gaia_photometry: pd.DataFrame, idstr: str = "ID", magstr: str = "MAG_AUTO_NORM", magerrstr: str = "MAGERR_AUTO", datestr: str = "DATE-OBS") -> dict[str, pd.DataFrame]:
    """
    Parse the Gaia Epoch photometry data to a format that can be used by the optimizer.
    :param gaia_photometry: The Gaia Epoch photometry data.
    :param magstr: The name that will be assigned to the magnitude column.
    :param magerrstr: The name that will be assigned to the magnitude error column.
    :return: The parsed Gaia photometry data in a dictionary with the filter as the key.
    """
    gaia_photometry = gaia_photometry[(gaia_photometry['rejected_by_photometry'] == 'false')
                                      & (gaia_photometry['rejected_by_variability'] == 'false')]
    gaia_photometry[magstr] = gaia_photometry['mag']
    gaia_photometry[magerrstr] = flux_to_magnitude_error(
        gaia_photometry['flux'], gaia_photometry['flux_error'])
    gaia_photometry[datestr] = bjd_tcb_to_mjd(gaia_photometry['time'])
    gaia_photometry[idstr] = gaia_photometry['source_id']
    gaia_photometry_data = {}
    for filter_key in gaia_photometry['band'].unique():
        parsed_filter_key = str(filter_key)
        gaia_photometry_data[parsed_filter_key] = gaia_photometry[gaia_photometry['band'] == filter_key]
        gaia_photometry_data[parsed_filter_key] = gaia_photometry_data[parsed_filter_key][[
            idstr, datestr, magstr, magerrstr]].copy()
    return gaia_photometry_data


def get_gaia_photometry(star_id: str) -> pd.DataFrame:
    """
    Get the Gaia epoch photometry data for a given star
    using the Gaia catalog data with astroquery.
    :param star_id: The ID of a star, which can be a Gaia ID 
        or a valid Simbad name or identifier.
    :return: The Gaia photometry data for the given star.
    """
    if not star_id.startswith('GAIA'):
        gaia_id = get_star_gaia_id(star_id)
    else:
        gaia_id = star_id.split('_')[-1]
    datalink: dict = Gaia.load_data(
        ids=gaia_id, retrieval_type='EPOCH_PHOTOMETRY', valid_data=True, format='csv')
    data_table: Table = datalink[next(iter(datalink))][0]
    return data_table.to_pandas()


def get_gaia_period(star_id: str) -> float:
    """
    Get the period of a star using the astroquery Gaia service.
    :param star_id: The ID of a star, which can be a Gaia ID 
        or a valid Simbad name or identifier.
    """
    if not star_id.startswith('GAIA'):
        gaia_id = get_star_gaia_id(star_id)
    else:
        gaia_id = star_id.split('_')[-1]
    query = f"""
        SELECT rrl.pf AS period
        FROM gaiadr3.vari_rrlyrae AS rrl
        WHERE rrl.source_id = {gaia_id}
    """
    job = Gaia.launch_job(query)
    result = job.get_results()
    if len(result) == 0:
        raise ValueError(f'No Gaia period data found for star {star_id}')
    return float(result['period'][0])


def filter_photometry_data(photometry_data: dict[str, pd.DataFrame], id_filename: str) -> dict[str, pd.DataFrame]:
    """
    Filter our photometry data to discard data from other stars.
    :param photometry_data: The photometry data for a given star.
    :param id_filename: The filename of the file with the star ID.
    :return: The filtered photometry data for the given star.
    """
    if os.path.exists(id_filename):
        varid = Table.read(id_filename, hdu=1)[0][0]
        pd.options.mode.chained_assignment = None
        filtered_data = {}
        for key in photometry_data:
            if 'Gaia' not in key:
                filtered_data[key] = photometry_data[key].loc[photometry_data[key].ID == varid, :]
            else:
                filtered_data[key] = photometry_data[key]
        return filtered_data
    return photometry_data


def get_star_photometry(photometry_path: str, star_id: str | None = None, idstr: str = "ID", magstr: str = "MAG_AUTO_NORM", magerrstr: str = "MAGERR_AUTO", datestr: str = "DATE-OBS") -> dict[str, pd.DataFrame]:
    """
    Get the prepared photometry data for a given star as a dict of dataframes for each filter.
    :param photometry_path: The path to the photometry data for the given star. 
    :return: The photometry data for the given star.
    """
    if star_id is None:
        star_id = Path(photometry_path).stem

    photometry_data = {}
    for photometry_file in os.listdir(photometry_path):
        if photometry_file.startswith('cat_'):
            filename_components = photometry_file.split('.')[0].split('_')
            if filename_components[-1] == 'final':
                band = filename_components[-2]
            else:
                band = filename_components[-1]
            photometry_data[band] = prepareTable(
                photometry_path+'/'+photometry_file, 1)
    gaia_photometry = parse_gaia_photometry(
        get_gaia_photometry(star_id), idstr, magstr, magerrstr, datestr)
    for band in gaia_photometry:
        photometry_data['Gaia-' + band] = gaia_photometry[band]

    photometry_data = filter_photometry_data(
        photometry_data, photometry_path+"/finalIDs_" + star_id+".fits")
    return photometry_data
