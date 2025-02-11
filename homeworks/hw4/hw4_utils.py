import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator, LogLocator
from scipy.optimize import curve_fit

def list_data_columns(df):
    for i in df.columns:
        print(i)

def gaia_query(ra_hex,dec_hex,radius,show=False,cons_scale=2,load_data=True):
    """
    Querries the Gaia database for stars in a cluster.

    Parameters:
        ra_hex (str): Right Ascension in hex
        dec_hex (str): Declination in hex
        radius (float): Radius of the cluster
        row_limit (int): Limit the number of rows returned
        show (bool): Display the data
        cons_scale (int): Scale the radius by this factor
    """

    if load_data:
        df = pd.read_csv('cluster_data.csv')

        # Convert the data to a astropy table
        cluster_data = Table.from_pandas(df)

        if show:
            cluster_data.pprint(max_lines=10,max_width=130)

        return cluster_data
    else:
        # Set Gaia data release
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1
        # Make a skycoord object
        coord = coord = SkyCoord(ra_hex, dec_hex, frame='icrs')  

        # Querry the gaia database for the cluster stars
        cluster_data = Gaia.query_object_async(coordinate=coord,radius = radius*cons_scale)

        if show:
            cluster_data.pprint(max_lines=10,max_width=130)
        return cluster_data


def get_gaia_two_body_orbit_cat(load_data=True,show=False,trimmed=False):
    """
    Get the Gaia data for the two body orbit catalog.
    """
    if load_data:
        if trimmed:
            cat = pd.read_csv('data2body_trimmed.csv')

        else:
            cat = pd.read_csv('data2body.csv')

        if show:
            print(cat)

    else:
        # Define your ADQL query
        query = """
        SELECT * FROM gaiadr3.nss_two_body_orbit
        """

        # Execute the query
        job = Gaia.launch_job_async(query)
        results = job.get_results()

        # Print the results
        if show:
            print(results)

        cat = results.to_pandas()

        # Save the results to a csv file
        cat.to_csv('data2body.csv')

    return cat