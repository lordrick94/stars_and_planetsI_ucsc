import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from astroquery.gaia import Gaia
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator, LogLocator
from scipy.optimize import curve_fit
import emcee
from scipy.optimize import minimize

from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

from dustmaps.bayestar import BayestarQuery
from dustmaps.config import config



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


def get_gaia_two_body_orbit_cat(load_data=True,show=False,processed=False,random=False):
    """
    Get the Gaia data for the two body orbit catalog.
    """
    if load_data:
        if processed:
            cat = pd.read_csv('hw4_cat_fit.csv')

        else:
            cat = pd.read_csv('data2body.csv')

        if show:
            print(cat)




    else:
        if random:
            # Getting a random stars from the Gaia database dr3
            query = """
            SELECT *
            FROM gaiadr3.gaia_source
            WHERE random_index BETWEEN 0 AND 200000
            """

            job = Gaia.launch_job_async(query)
            results = job.get_results()

            # Print the results
            if show:
                print(results)

            cat = results.to_pandas()

            # Save the results to a csv file
            cat.to_csv('random_stars.csv')
            pass

        else:
            # Define your ADQL query
            query = """
                    SELECT * FROM gaiadr3.nss_two_body_orbit 
                    """

            #TODO: Join with the gaia source table to get the parallax and CMD stuff

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


def get_cross_match_cat():
        # Define your ADQL query
        query = """
        SELECT nss.*,
            gs.ra, gs.dec, gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
            gs.phot_g_mean_flux_error, gs.phot_bp_mean_flux_error, gs.phot_rp_mean_flux_error, gs.bp_rp, gs.ag_gspphot, gs.ebpminrp_gspphot
        FROM gaiadr3.nss_two_body_orbit AS nss
        JOIN gaiadr3.gaia_source AS gs USING(source_id)
        """

        #TODO: Join with the gaia source table to get the parallax and CMD stuff

        # Execute the query
        job = Gaia.launch_job_async(query)
        results = job.get_results()
        cat = results.to_pandas()

        # Save the results to a csv file
        cat.to_csv('cross_match_table.csv')

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
    # new_cat = new_cat[rm_cols]

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

def get_fit_params(cat,load_data=True):
    """
    Fit the Thiele-Innes constants to the data.
    """
    if load_data:
        cat_fit = pd.read_csv('fit_params.csv')
    else:
        # Fit the Thiele-Innes constants for each row in the catalog
        fit_params = cat.apply(fit_for_params, axis=1)

        # Convert the fit parameters to a DataFrame
        fit_params_df = pd.DataFrame(list(fit_params),index=cat.index)

        # Combine the original catalog with the fit parameters
        cat_fit = pd.concat([cat, fit_params_df], axis=1)

    return cat_fit



def calculate_extinction(ra, dec, parallax, bayestar=None):
    """
    Calculate extinction for Gaia BP, G, and RP bands using the Bayestar 3D dust map.

    Parameters:
        ra (float): Right Ascension in degrees
        dec (float): Declination in degrees
        parallax (float): Parallax in milliarcseconds (mas)

    Returns:
        dict: Extinction values A_BP, A_G, A_RP
    """


    if parallax <= 0:
        raise ValueError("Parallax must be positive.")

    # Convert parallax (mas) to distance (parsecs)
    distance_pc = 1000.0 / parallax
    distance = Distance(distmod=5 * np.log10(distance_pc) - 5)

    # Convert to Galactic coordinates
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distance, frame='icrs').galactic

    # Query the Bayestar dust map for E(B-V)
    ebv = bayestar(coord)

    # Convert E(B-V) to extinction in Gaia bands
    if ebv is None:
        return {'A_BP': np.nan, 'A_G': np.nan, 'A_RP': np.nan}

    # Coefficients for Schlafly et al. (2018) extinction conversion for Gaia bands
    coeff_bp = 3.14
    coeff_g = 2.27
    coeff_rp = 1.61

    # Calculate extinction
    a_bp = coeff_bp * ebv
    a_g = coeff_g * ebv
    a_rp = coeff_rp * ebv

    return {'A_BP': a_bp, 'A_G': a_g, 'A_RP': a_rp}

def get_extinction(cat,load_data=True,save_data=None,random_stars=False):
    """
    Calculate the extinction for the catalog.
    """
    if load_data:
        if random_stars:
            cat_ext = pd.read_csv('random_stars_ext.csv')
        else:
            cat_ext = pd.read_csv('extinction.csv')
    else:
        # Set up the dustmaps cache directory
        config['data_dir'] = '~/.dustmaps'

        # Initialize the 3D dust map
        bayestar = BayestarQuery(version='bayestar2019')
        # Calculate extinction for each row in the catalog
        extinctions = cat.apply(lambda row: calculate_extinction(row['ra'], row['dec'], row['parallax'],bayestar=bayestar), axis=1)

        # Convert the extinction values to a DataFrame
        extinctions_df = pd.DataFrame(list(extinctions), index=cat.index)

        # Combine the original catalog with the extinction values
        cat_ext = pd.concat([cat, extinctions_df], axis=1)

    
    final_cat = cat_ext.copy()
    # Make corrections
    final_cat.loc[:,'G_corr'] = final_cat['phot_g_mean_mag'] - final_cat['A_G']
    final_cat.loc[:,'BP-RB_corr'] = final_cat['bp_rp'] - (final_cat['A_BP'] - final_cat['A_RP'])

    if save_data is not None:
        final_cat.to_csv(save_data)
    return cat_ext

# Make a color magnitude diagram
def plot_color_magnitude_diagram(dfs,ssize=5,
                                 colors=['r','b','g'],
                                 labels=['Cluster Stars 1','Clusters Stars 2','Clusters Stars 3'],
                                 log_x=False):
    # Extract necessary columns (g band magnitude and bp-rp color)
    abs_mag_col = 'G_corr'
    color_col = 'BP-RB_corr'



    # Create the plot
    fig, ax = plt.subplots(figsize=(16/2., 9/2.))

    def make_scatter(df,ssize=5,plt_color='r',lbl='Cluster Stars'):    
        scatter = ax.scatter(
        df[color_col],
        df[abs_mag_col], 
        s=ssize, 
        c=plt_color, 
        alpha=0.4,  # Increase transparency
        edgecolor='k',  # Optional for clarity
        label=lbl,
        cmap='plasma',
        norm=LogNorm()  # Apply logarithmic normalization
            )
        
        return scatter
    
    
    # Plot the cluster stars
    for r, color,l in zip(dfs, colors,labels):
        s = make_scatter(r,ssize=ssize,plt_color=color,lbl=l)
    # # Add a color bar to show the colormap scale
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label(r'Parallax Error', fontsize=12)
        
    ax.set_xlabel('BP - RP [mag]', fontsize=12)
    ax.set_ylabel('Absolute Magnitude G band [mag]', fontsize=12)
    ax.set_title('Color-Magnitude Diagram', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    if log_x:
        ax.set_xscale('log')


    ax.legend()

    plt.savefig(f'cmd_plot.png', dpi=400, bbox_inches='tight')

    plt.show()

    return fig,ax
    

###############################################################################################
# MESA STUFF

def read_data(filepath):
    """
    Reads MESA's .data files and returns it as a pandas DataFrame.
    
    Parameters:
        filepath (str): Path to the history.data file.
    
    Returns:
        pd.DataFrame: Processed DataFrame with the relevant columns.
    """
    # Skip the first 5 rows (header info) and read column names from the 6th row
    data = pd.read_csv(filepath, sep=r'\s+', skiprows=5)
    return data


def plot_property_vs_age(df,
                        property_col:str, 
                        ax=None, 
                        label='Brown Dwarf', 
                        color='blue',
                        lstyl='-', 
                        ylabl=r'Log$(L)$ $(erg/s$)',
                        get_abundance=False,
                        ab_ssize=5,
                        add_abundance_lbls=True):
    """
    Plots the property column vs age profile of a star.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the profile data.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axes are created.
        label (str, optional): Label for the plot.
        color (str, optional): Color for the plot.
    """
    # Extract the necessary columns
    star_age = df['star_age']
    logL = df[property_col]
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Plot the data    
    ax.plot(star_age, logL, color=color, label=label, linestyle=lstyl)

    if get_abundance:
        abundance_cols = ['center_h2','center_li7']
        if add_abundance_lbls:
            abundance_labels = [r'$H_2$ abundance reduced by a factor 100',r'$Li_7$ abundance reduced by a factor of 100']
        else:
            abundance_labels = [None,None]
        abundance_colors = ['r','b']
        idxs = [get_abundance_points(df,abundance_col) for abundance_col in abundance_cols]

        for abundance_col,abundance_label,abundance_color,idx in zip(abundance_cols,abundance_labels,abundance_colors,idxs):
            # Make a scatter point at abundance_age with corresponding value in df[property_col]
            if idx is not None:
                ax.scatter(df['star_age'].iloc[idx],
                           df[property_col].iloc[idx],
                           color=abundance_color,
                           label=abundance_label,
                           marker='*',s=ab_ssize,
                           edgecolors='k',
                           alpha=0.5)
            
            else:
                continue



    # Set the font dictionaries (for plot title and axis titles)
    labl_fontdict = {'family': 'serif', 'color':  'k', 'weight': 'normal','size': 12}

    # Label the axes
    ax.set_xlabel(r'Star Age (Years)', fontsize=12)
    ax.set_ylabel(ylabl,fontdict=labl_fontdict)
    ax.tick_params(axis='both', which='major', color=labl_fontdict['color'],labelsize=labl_fontdict['size'],labelcolor=labl_fontdict['color'])
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show the plot
    if ax is None:
        plt.savefig('Luminosity_vs_starage.png', dpi=300)
        plt.show()

def get_abundance_points(df, abundance_col,verbose=False):
    """
    Get the age where the abundance has decreased by a factor of 100.
    """
    abundances = df[abundance_col]
    age = df['star_age']

    # Get the initial abundance
    init_abundance = abundances.iloc[0]  # Use iloc to ensure first value is taken correctly

    # Get the reduced abundance
    reduced_abundance = init_abundance / 100

    # Find indices where abundance has dropped below reduced_abundance
    valid_indices = age[abundances < reduced_abundance]

    if valid_indices.empty:
        if verbose:
            print(f'No abundance decrease by a factor of 100 for {abundance_col}')
        return None

    # Get the index where the abundance has first dropped below the threshold
    idx = valid_indices.idxmin()

    return idx



    
