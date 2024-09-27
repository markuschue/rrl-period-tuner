import argparse


def parse_cli_args(description: str) -> tuple:
    """
    Parse the command line arguments.
    :return: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--photometry_path', type=str,
                        help='The directory containing the photometry data', required=True)
    parser.add_argument('-n', '--name', type=str,
                        help='The name of the star to analyze. If not specified, it will try to be inferred from the photometry path',
                        required=False)

    args = parser.parse_args()

    photometry_path: str = args.photometry_path
    if photometry_path.endswith("/") or photometry_path.endswith("\\"):
        photometry_path = photometry_path[:-1]
    if args.name is not None:
        star_id: str = args.name
    else:
        star_id = photometry_path.split("/")[-1]
        if '_' in star_id:
            star_id = star_id.split("_")[1]
        if 'gaia' in photometry_path.lower():
            star_id = 'Gaia DR3 ' + star_id

    return photometry_path, star_id
