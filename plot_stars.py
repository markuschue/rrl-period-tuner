from os import listdir

import matplotlib.pyplot as plt
import pandas as pd


def plot_light_curves(file_path: str, star_lookup_path: str, category: str, save_data: bool = False):
    data = pd.read_csv(file_path)
    star_lookup = pd.read_csv(star_lookup_path)

    # Filter out data points that were rejected by photometry or variability
    filtered_data = data[(data['rejected_by_photometry'] == False) & (
        data['rejected_by_variability'] == False)]

    source_ids = filtered_data['source_id'].unique()

    # The files only contain data for one source, so there should only be one source_id per file,
    # but we'll loop through them anyway for consistency
    for source_id in source_ids:
        star_data = filtered_data[filtered_data['source_id'] == source_id]
        star_lookup_data = star_lookup[star_lookup['source_id'] == source_id]

        if star_lookup_data["ra"].values[0] > 270:
            continue

        bands = star_data['band'].unique()

        plt.figure(figsize=(10, 6))

        for band in bands:
            band_data = star_data[star_data['band'] == band]
            # Plot the light curve for each band, with points and thin lines that connect them
            plt.plot(band_data['time'], band_data['mag'], linestyle='-',
                     linewidth=0.5, alpha=0.7, label=f'Band {band} (line)')
            plt.scatter(band_data['time'], band_data['mag'],
                        label=f'Band {band} (points)', s=10)

        star_values = f'{source_id}, {category}, {star_lookup_data["rrl.pf"].values[0]},{
            star_lookup_data["phot_g_mean_mag"].values[0]}, {star_lookup_data["ra"].values[0]}, {
            star_lookup_data["dec"].values[0]}\n'
        print(star_values)
        if save_data:
            # Save the printed values to a CSV file
            with open('output.csv', 'a') as file:
                # Headers: source_id, category, period, mean_magnitude
                if file.tell() == 0:
                    file.write(
                        'source_id, category, period, mean_magnitude, ra, dec\n')
                file.write(f'{source_id}, {category}, {star_lookup_data["rrl.pf"].values[0]},{
                    star_lookup_data["phot_g_mean_mag"].values[0]}, {star_lookup_data["ra"].values[0]}, {
                        star_lookup_data["dec"].values[0]}\n')

        plt.gca().invert_yaxis()
        plt.title(f'Light Curve for Source ID: {source_id}')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # The data folder is organized in the following way:
    # data/candidates/
    # ├── HPHM      --- High Period High Magnitude
    # ├── HPLM      --- High Period Low Magnitude
    # ├── LPHM      --- Low Period High Magnitude
    # └── LPLM      --- Low Period Low Magnitude

    DATA_DIR = 'data/candidates/'
    categories = listdir(DATA_DIR)
    for category in categories:
        print(category)
        filenames = listdir(f'{DATA_DIR}/{category}')
        for file in filenames:
            plot_light_curves(f'{DATA_DIR}/{category}/{file}',
                              'data/star_lookup.csv', category)
