from os import listdir

import pandas as pd

from utils import flux_to_magnitude_error


def translate_band(band: str) -> str:
    """
    Translate the band from the Gaia format to the LCV format.
    :param band: The band in the Gaia format.
    :return: The band in the LCV format.
    """
    if band == 'G':
        return '3'
    if band == 'BP':
        return '4'
    if band == 'RP':
        return '5'
    return '0'


def format_date(bjd: float) -> str:
    """
    Format the BJD date to the IDL format in JD.
    :param bjd: The BJD value.
    :return: The JD value as a string, with the year and the rest of the date separated by a space.
    """
    date = bjd + 2455197.5
    date = str(date)
    return date[:4] + ' ' + date[4:]


def csv_to_lcv(path: str, output_path: str | None = None):
    """
    Convert a CSV file to an LCV file.
    :param path: The path to the CSV file.
    """
    gaia_star = pd.read_csv(path)
    lcv_path = path.replace(
        '.csv', '.lcv') if not output_path else output_path
    with open(lcv_path, 'w') as output:
        for _, row in gaia_star.iterrows():
            if row['rejected_by_photometry'] or row['rejected_by_variability']:
                continue
            output.write(
                f"{row['mag']}\t{flux_to_magnitude_error(row['flux'], row['flux_error'])}\t{translate_band(row['band'])}\t0\t{format_date(row['time'])}\t0\t0\t0\n")


if __name__ == '__main__':
    DATA_DIR = 'final_candidates'
    OUT_DIR = 'final_candidates_lcv'
    categories = listdir(DATA_DIR)
    for category in categories:
        for file in listdir(f'{DATA_DIR}/{category}'):
            lcv_filename = file.replace('.csv', '.lcv')
            if file.endswith('.csv'):
                csv_to_lcv(f'{DATA_DIR}/{category}/{file}',
                           f'{OUT_DIR}/{category}/{lcv_filename}')
