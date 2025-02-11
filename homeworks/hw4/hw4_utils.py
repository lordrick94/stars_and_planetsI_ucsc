import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator, LogLocator
from scipy.optimize import curve_fit
import emcee
from scipy.optimize import minimize

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

def clean_catalog(cat,col_list=None):
    """
    Clean the catalog by removing rows with NaN values.
    """
    rm_cols = ['ra','dec', 'parallax_error', 'pmra', 'pmra_error',
       'pmdec', 'pmdec_error', 'a_thiele_innes', 'a_thiele_innes_error',
       'b_thiele_innes', 'b_thiele_innes_error', 'f_thiele_innes',
       'f_thiele_innes_error', 'g_thiele_innes', 'g_thiele_innes_error', 
       'period', 'period_error', 't_periastron',
       't_periastron_error', 'eccentricity', 'eccentricity_error']

    if col_list is not None:
        rm_cols = col_list

    for col in rm_cols:
        cat = cat.dropna(subset=[col])

    new_cat = cat.reset_index(drop=True)
    new_cat = new_cat[rm_cols]

    return new_cat


def fit_for_params(idf):
    # Given Thiele-Innes constants
    A_obs = idf['a_thiele_innes']
    B_obs = idf['b_thiele_innes']
    F_obs = idf['f_thiele_innes']
    G_obs = idf['g_thiele_innes']

        # Define the function to minimize (least squares error function)
    def fit_function(params):
        a0, omega, Omega, i = params

        # Convert angles to radians
        omega = np.radians(omega)
        Omega = np.radians(Omega)
        i = np.radians(i)

        # Compute the modeled Thiele-Innes constants
        A_calc = a0 * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
        B_calc = a0 * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
        F_calc = -a0 * (np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.sin(Omega) * np.cos(i))
        G_calc = -a0 * (np.sin(omega) * np.sin(Omega) - np.cos(omega) * np.cos(Omega) * np.cos(i))

        # Compute squared error
        error = (A_calc - A_obs)**2 + (B_calc - B_obs)**2 + (F_calc - F_obs)**2 + (G_calc - G_obs)**2

        return error

    # Initial guess for [a0, omega (deg), Omega (deg), i (deg)]
    initial_guess = [1.0, 45, 45, 45]

    # Perform the optimization
    result = minimize(fit_function, initial_guess, method='Nelder-Mead')

    # Extract the best-fit parameters
    a0_fit, omega_fit, Omega_fit, i_fit = result.x

    # Convert angles back to degrees
    omega_fit_deg = omega_fit
    Omega_fit_deg = Omega_fit
    i_fit_deg = i_fit

    # Display results
    a0_fit, omega_fit_deg, Omega_fit_deg, i_fit_deg

    result = {'a0_fit':a0_fit,'omega_fit_deg':omega_fit_deg,'Omega_fit_deg':Omega_fit_deg,'i_fit_deg':i_fit_deg}

    return result

def get_fit_params(cat):
    """
    Fit the Thiele-Innes constants to the data.
    """
    # Fit the Thiele-Innes constants for each row in the catalog
    fit_params = cat.apply(fit_for_params, axis=1)

    # Convert the fit parameters to a DataFrame
    fit_params_df = pd.DataFrame(list(fit_params),index=cat.index)

    # Combine the original catalog with the fit parameters
    cat_fit = pd.concat([cat, fit_params_df], axis=1)

    return cat_fit

