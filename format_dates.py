import os
import re

import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

from utils.data_utils import get_star_photometry


def get_ra_dec(star_id: str, data_csv_path: str) -> tuple[float, float]:
    data = pd.read_csv(data_csv_path)
    data["source_id"] = data["source_id"].astype(str)
    star_data = data[data["source_id"] == star_id]
    return star_data["ra"].values[0], star_data["dec"].values[0]


def format_dates(data: dict[str, pd.DataFrame], ra: float, dec: float) -> dict[str, pd.DataFrame]:
    # Baryocentric Julian Day
    # header = fits.open(median["V"].FILENAME.iloc[0],hdu=0)[0].header
    earthcoord = EarthLocation(
        lat='28 17 58.8 N', lon='16 30 39.7 W', height=2381.25)
    # earthcoord = EarthLocation(lat=header["SITELAT"],lon=header["SITELONG"],height=2381.25)

    for key in data:
        coord = SkyCoord(ra=ra, dec=dec, unit='deg', equinox="J2000")
        bjd = Time(data[key]["DATE-OBS"], format='mjd').light_travel_time(skycoord=coord,
                                                                          kind='barycentric', location=earthcoord)
        data[key].loc[:, "DATE-OBS"] = (Time(data[key].loc[:, "DATE-OBS"], format='mjd') +
                                        bjd).to_value('mjd').astype('float64')

    return data


def save_to_file(data: dict[str, pd.DataFrame], photometry_folder: str):
    for key in data:
        if 'Gaia' not in key:
            tab = Table.from_pandas(data[key])
            tab.write(f'data/photometry/{photometry_folder}/cat_{
                photometry_folder}_{key}.fits', overwrite=True)


if __name__ == '__main__':
    for star_photometry_folder in os.listdir('data/photometry'):
        star_id = star_photometry_folder.split('/')[-1]
        if star_id.endswith('/'):
            star_id = star_id[:-1]
        if 'GAIA' in star_photometry_folder:
            ra, dec = get_ra_dec(star_id.split(
                '_')[-1], 'data/star_lookup.csv')
            star_id = 'Gaia DR3 ' + star_id.split('_')[-1]
        else:
            ra, dec = get_ra_dec(star_id, 'data/ra_dec.csv')
            star_id = re.sub(r'([A-Z])([a-z]+$)', r' \1\2',
                             star_id.split('_')[-1])
        star_photometry_data = get_star_photometry(
            'data/photometry/' + star_photometry_folder, star_id)
        star_photometry_data = format_dates(star_photometry_data, ra, dec)
        # save_to_file(star_photometry_data, star_photometry_folder)
