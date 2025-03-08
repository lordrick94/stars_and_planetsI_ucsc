import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import seaborn as sns

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

def angular_orbital_elements(idf):
    """
    Calculate the angular orbital elements for the two-body orbit catalog.
    """
    df = idf.copy()
    # Given Thiele-Innes constants
    A_obs = df['a_thiele_innes']
    B_obs = df['b_thiele_innes']
    F_obs = df['f_thiele_innes']
    G_obs = df['g_thiele_innes']

    df.loc[:,'u_obs'] = (A_obs**2 + B_obs**2 + F_obs**2 + G_obs**2 )/2
    df.loc[:,'v_obs'] = (A_obs*G_obs - B_obs*F_obs)
    df.loc[:,'a0_obs'] = np.sqrt(df['u_obs'] + np.sqrt(df['u_obs']**2 - df['v_obs']**2))

    df.loc[:,'m_f'] = (df['a0_obs']**3)*(df['parallax']**-3)*(df['period']/365.25)**-2

    df.loc[:,'d_in_au'] = [(((1000/x)*u.pc).to(u.au)).value for x in df['parallax']]

    df.loc[:,'ad_in_au'] = df['a0_obs']*df['d_in_au']

    # Calculate M_2 for M_1 = 1 solar mass
    df.loc[:,'m2_1'] = [get_dark_mass_companion(1,x) for x in df['m_f']]

    return df

def get_binaries(load=False):
    if load:
        return pd.read_csv('binaries.csv')
    
    else:
        df1 = pd.read_csv('cross_match_table.csv')
        df2 = pd.read_csv('extinction.csv')

        # merge the two dataframes but select only one column [goodness_of_fit] from df1
        df = pd.merge(df2,df1[['goodness_of_fit','source_id']], on='source_id', how='inner')

        good_df = df[df['goodness_of_fit'] < 5]

        a0_df = angular_orbital_elements(good_df)

        a0_df.to_csv('binaries.csv',index=False)

        return a0_df


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


def get_dark_mass_companion(m_f,m_1):
    """
    Solve the cubic equation:
        m2^3 - m_f*m2^2 - 2*m_f*m1*m2 - m_f*m1^2 = 0
    for m2, given m_f and m1.
    
    Parameters:
        m_f (float): known parameter.
        m1 (float): known parameter.
        
    Returns:
        m2 (float): The positive real solution for m2.
    """
 
    # Define the coefficients of the cubic equation.
    coeffs = [1, -m_f, -2 * m_f * m_1, -m_f * m_1**2]
    
    # Compute all roots of the polynomial.
    roots = np.roots(coeffs)
    
    # Filter out only the real, positive roots.
    real_positive_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    if not real_positive_roots:
        raise ValueError("No positive real roots found for m2.")
    
    # Return the smallest positive root (or choose based on your application).
    return min(real_positive_roots)
    

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

def plot_center_T_vs_center_Rho(df1, df2, df3, df4, labels, colors, 
                                ZAMS_model_numbers, TAMS_model_numbers, 
                                ZAMS_text_offset=[-0.1,-0.1,-0.1,-0.1], TAMS_text_offset=[0.06,-0.1,-0.1,-0.1],
                                ZAMS_text_yoffset=[-0.1,-0.1,-0.1,-0.1], TAMS_text_yoffset=[0.06,-0.1,-0.1,-0.1]):
    """
    Plots the central temperature vs central density profile for four datasets on a single plot.
    Also marks the Zero-Age Main Sequence (ZAMS) and Terminal-Age Main Sequence (TAMS) points.

    Parameters:
        df1, df2, df3, df4 (pd.DataFrame): DataFrames containing the profile data.
        labels (list of str): Labels for each dataset.
        colors (list of str): Colors for each dataset.
        ZAMS_model_numbers (list of int or None): Model numbers for ZAMS points in each dataset.
        TAMS_model_numbers (list of int or None): Model numbers for TAMS points in each dataset.
        ZAMS_text_offset (list of float, optional): Offset for ZAMS annotation text.
        TAMS_text_offset (list of float, optional): Offset for TAMS annotation text.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    dataframes = [df1, df2, df3, df4]
    
    for df, label, color, ZAMS_model_number, TAMS_model_number, zt, tt,zy,ty in zip(
        dataframes, 
        labels, 
        colors, 
        ZAMS_model_numbers, 
        TAMS_model_numbers, 
        ZAMS_text_offset, 
        TAMS_text_offset,
        ZAMS_text_yoffset,
        TAMS_text_yoffset):
        
        log_center_T = df['log_center_T']
        log_center_Rho = df['log_center_Rho']
        
        ax.plot(log_center_T, log_center_Rho, color=color, label=label)

        # Mark ZAMS and TAMS points if provided
        if ZAMS_model_number is not None and ZAMS_model_number in df['model_number'].values:
            ZAMS_T = df.loc[df['model_number'] == ZAMS_model_number, 'log_center_T'].values[0]
            ZAMS_Rho = df.loc[df['model_number'] == ZAMS_model_number, 'log_center_Rho'].values[0]
            ax.scatter(ZAMS_T, ZAMS_Rho, color='red', marker='*', s=100, edgecolor='black', label='ZAMS' if label == labels[0] else "", alpha=0.7)
            ax.annotate('ZAMS', xy=(ZAMS_T, ZAMS_Rho), xytext=(ZAMS_T + zt, ZAMS_Rho + zy),
                        fontsize=10, ha='right', va='bottom', color='red',
                        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5))

        if TAMS_model_number is not None and TAMS_model_number in df['model_number'].values:
            TAMS_T = df.loc[df['model_number'] == TAMS_model_number, 'log_center_T'].values[0]
            TAMS_Rho = df.loc[df['model_number'] == TAMS_model_number, 'log_center_Rho'].values[0]
            ax.scatter(TAMS_T, TAMS_Rho, color='green', marker='*', s=100, edgecolor='black', label='TAMS' if label == labels[0] else "", alpha=0.7)
            ax.annotate('TAMS', xy=(TAMS_T, TAMS_Rho), xytext=(TAMS_T + tt, TAMS_Rho + ty),
                        fontsize=10, ha='right', va='bottom', color='green',
                        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5))

    ax.set_xlabel(r'Log$(T_{c})$ (K)', fontsize=14)
    ax.set_ylabel(r'Log$(\rho_{c})$ (g/cm$^3$)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('center_T_vs_center_Rho_combined.png', dpi=300)
    plt.show()





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

def plot_mesa_hr_diagram(data, 
                     output_file=None, 
                    ZAMS_model_number=None, 
                    TAMS_model_number=None, 
                    min_model_number=None, 
                    max_model_number=None, 
                    middle_model_number=None,
                    ZAMS_text_offset=-0.1,
                    TAMS_text_offset=0.06,
                    fig_title=None):
    """
    Plots the Hertzsprung-Russell diagram using MESA's history data.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing MESA's history data.
        output_file (str): Optional. Path to save the plot as an image.
    """
    if max_model_number is not None:
        # # Filter data based on model number range if provided
        data = data[(data['model_number'] >= min_model_number) & (data['model_number'] <= max_model_number)]

        data = data.reset_index(drop=True)

    # Extract necessary columns: log_Teff and log_L
    log_Teff = data['log_Teff']
    log_L = data['log_L']

    # center_h1 as the color
    center_h1 = data['center_h1']


    if ZAMS_model_number and TAMS_model_number:
        # Get the ZAMS and TAMS values to mark on the scatter
        ZAMS_log_Teff = data.loc[data['model_number'] == ZAMS_model_number, 'log_Teff'].values[0]
        ZAMS_log_L = data.loc[data['model_number'] == ZAMS_model_number, 'log_L'].values[0]


        TAMS_log_Teff = data.loc[data['model_number'] == TAMS_model_number, 'log_Teff'].values[0]
        TAMS_log_L = data.loc[data['model_number'] == TAMS_model_number, 'log_L'].values[0]
        
        # Create the HR diagram (Note: HR diagrams are typically flipped for log_Teff)
        plt.figure(figsize=(10, 8))

        
        # Annotate the ZAMS and TAMS points with arrows
        if ZAMS_model_number and ZAMS_log_Teff is not None and ZAMS_log_L is not None:
            plt.scatter(ZAMS_log_Teff, ZAMS_log_L, color='red', marker='*', s=200, edgecolor='black', label='ZAMS', alpha=0.5)
            plt.annotate('ZAMS', xy=(ZAMS_log_Teff, ZAMS_log_L), xytext=(ZAMS_log_Teff + ZAMS_text_offset, ZAMS_log_L + 0.00),
                        fontsize=12, ha='right', va='bottom', color='red',
                        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5))

        if TAMS_model_number and TAMS_log_Teff is not None and TAMS_log_L is not None:
            plt.scatter(TAMS_log_Teff, TAMS_log_L, color='green', marker='*', s=200, edgecolor='black', label='TAMS', alpha=0.5)
            plt.annotate('TAMS', xy=(TAMS_log_Teff, TAMS_log_L), xytext=(TAMS_log_Teff + TAMS_text_offset, TAMS_log_L + 0.0),
                        fontsize=12, ha='right', va='bottom', color='green',
                        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5))
            

    if middle_model_number:
        middle_log_Teff = data.loc[data['model_number'] == middle_model_number, 'log_Teff'].values[0]
        middle_log_L = data.loc[data['model_number'] == middle_model_number, 'log_L'].values[0]
        plt.scatter(middle_log_Teff, middle_log_L, color='blue', marker='*', s=200, edgecolor='black', label='Mid MS Point', alpha=0.5)
        plt.annotate('Mid MS Point', xy=(middle_log_Teff, middle_log_L), xytext=(middle_log_Teff - 0.02, middle_log_L + 0.0),
                        fontsize=12, ha='right', va='bottom', color='blue',
                        arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.5))
    
    scatter = plt.scatter(log_Teff, log_L, c=center_h1, cmap='viridis', edgecolor='k', s=50, alpha=0.7, label='Stellar Evolution Track')
    plt.gca().invert_xaxis()  # Flip the x-axis for decreasing temperature
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Center H1 Fraction', fontsize=12)
    
    # Label the axes
    plt.xlabel(r'Log$(T_{eff})$ (K)', fontsize=14)
    plt.ylabel(r'Log Luminosity (log$\frac{L}{L_{\odot}}$)', fontsize=14)

    if fig_title:
        plt.title(fig_title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def plot_mass_profiles(zams_cno, zams_pp, tams_cno, tams_pp, 
                       fig_title=None,
                       output_file=None,
                       log_eps_nuc=False):
    """
    Plots the interior structure (density, temperature, and nuclear energy generation rate)
    as a function of mass coordinate for two evolutionary stages (ZAMS and TAMS).
    
    For each stage, the profiles from two different reaction networks (CNO and PP) are overplotted.
    
    Parameters:
        zams_cno (pd.DataFrame): ZAMS model using the CNO network.
        zams_pp  (pd.DataFrame): ZAMS model using the PP network.
        tams_cno (pd.DataFrame): TAMS model using the CNO network.
        tams_pp  (pd.DataFrame): TAMS model using the PP network.
        
    Expected DataFrame columns:
        'mass'    : Mass coordinate (in solar masses)
        'logRho'  : Logarithm of density
        'logT'    : Logarithm of temperature
        'eps_nuc' : Nuclear energy generation rate
    """
    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex='col')
    
    # Define labels for each row
    row_labels = [r'Log($\rho$)', r'Log($T$)', r'$\epsilon_{nuc}$']
    
    # Define a mapping for which column in the DataFrame to plot for each row:
    # Row 0: 'logRho', Row 1: 'logT', Row 2: 'eps_nuc'
    data_keys = ['logRho', 'logT', 'eps_nuc']
    
    # Define colors/styles for the two networks
    styles = {
        'CNO': {'color': 'blue', 'linestyle': '-'},
        'PP' : {'color': 'red',  'linestyle': '--'}
    }
    if log_eps_nuc:
        zams_cno = make_log_eps_nuc_col(zams_cno)
        zams_pp = make_log_eps_nuc_col(zams_pp)
        tams_cno = make_log_eps_nuc_col(tams_cno)
        tams_pp = make_log_eps_nuc_col(tams_pp)
        data_keys = ['logRho', 'logT', 'log_eps_nuc']
    
    # Define a helper function to plot on a given axis
    def plot_profiles(ax, df_cno, df_pp, key, ylabel):
        ax.plot(df_cno['mass'], df_cno[key],
                label='CNO', **styles['CNO'])
        ax.plot(df_pp['mass'], df_pp[key],
                label='PP', **styles['PP'])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
    
    # Plot ZAMS profiles in left column
    for i, key in enumerate(data_keys):
        plot_profiles(axs[i, 0], zams_cno, zams_pp, key, row_labels[i])
        if i == 2:
            axs[i, 0].set_xlabel('Mass (M$_{\odot}$)', fontsize=12)
    
    # Plot TAMS profiles in right column
    for i, key in enumerate(data_keys):
        plot_profiles(axs[i, 1], tams_cno, tams_pp, key, row_labels[i])
        if i == 2:
            axs[i, 1].set_xlabel('$M_r$ (M$_{\odot}$)', fontsize=12)


    # Add figure title and adjust layout
    fig.suptitle(fig_title, fontsize=16)
    
    plt.tight_layout()

    # Save
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def make_log_eps_nuc_col(df):
    df['log_eps_nuc'] = np.log10(df['eps_nuc'])
    return df


def plot_abundances(df1, df2, label1, label2,
                    output_file='abundances_vs_radius.png',
                    log_yscale=False,
                    log_xscale=False,
                    fig_title=None):
    """
    Plots the abundances of different elements in a star as a function of radius.
    
    Parameters:
        df1 (pd.DataFrame): DataFrame containing the profile data.
        df2 (pd.DataFrame): DataFrame containing the profile data.
    """

    def plot_element(radius, abundance, label, color, linestyle='-', marker=None, alpha=0.7):
        plt.plot(radius, abundance, color=color, label=label, linestyle=linestyle, marker=marker, alpha=alpha, linewidth=3)

    # Extract the necessary columns
    if log_xscale:
        radius1 = df1['logR']
        radius2 = df2['logR']
        custom_xlabel = r'Log(Radius [$R/R_{\odot}$])'
    else:
        radius1 = 10**df1['logR']
        radius2 = 10**df2['logR']
        custom_xlabel = r'Radius [$R/R_{\odot}$]'

    elements = {
        'H': ('h1', 'lightblue', 'blue'),
        'He': ('he4', 'palevioletred', 'crimson'),
        'C': ('c12', 'lightgreen', 'green'),
        'N': ('n14', 'moccasin','orange'),
        'O': ('o16', 'lightcoral', 'red'),
    }

    # Use seaborn style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 8))

    for i, (element, (column, color1,color2)) in enumerate(elements.items()):
        plot_element(radius1, df1[column], f'{element} {label1}', color1, linestyle='--')
        plot_element(radius2, df2[column], f'{element} {label2}', color2, linestyle='-')


    # Label the axes
    plt.xlabel(custom_xlabel, fontsize=14)
    if log_yscale:
        plt.yscale('log')
    plt.ylabel(r'mass fraction', fontsize=14)

    if fig_title:
        plt.title(fig_title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or show the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()






    