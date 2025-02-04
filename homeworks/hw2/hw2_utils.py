import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import astropy.constants as const
from matplotlib.colors import LogNorm

def plot_hr_diagram(data, 
                    output_file=None, 
                    ZAMS_model_number=None, 
                    TAMS_model_number=None, 
                    max_model_number=None, 
                    min_model_number=None, 
                    middle_model_number=None):
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
            plt.annotate('ZAMS', xy=(ZAMS_log_Teff, ZAMS_log_L), xytext=(ZAMS_log_Teff - 0.1, ZAMS_log_L + 0.00),
                        fontsize=12, ha='right', va='bottom', color='red',
                        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5))

        if TAMS_model_number and TAMS_log_Teff is not None and TAMS_log_L is not None:
            plt.scatter(TAMS_log_Teff, TAMS_log_L, color='green', marker='*', s=200, edgecolor='black', label='TAMS', alpha=0.5)
            plt.annotate('TAMS', xy=(TAMS_log_Teff, TAMS_log_L), xytext=(TAMS_log_Teff + 0.06, TAMS_log_L + 0.0),
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
    plt.title('Hertzsprung-Russell Diagram', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def read_data(filepath):
    """
    Reads MESA's .data files and returns it as a pandas DataFrame.
    
    Parameters:
        filepath (str): Path to the history.data file.
    
    Returns:
        pd.DataFrame: Processed DataFrame with the relevant columns.
    """
    # Skip the first 5 rows (header info) and read column names from the 6th row
    data = pd.read_csv(filepath, delim_whitespace=True, skiprows=5)
    return data


def plot_rho_t(df):
    """
    Plots the density vs temperature profile of a star.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the profile data.
    """
    # Extract the necessary columns
    logRho = df['logRho']
    logT = df['logT']
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(logRho, logT, color='blue', label='Density vs Temperature')

    # Annotate the star's center and surface
    plt.scatter(logRho[0], logT[0], color='red', marker='o', s=100, label='Surface')

    plt.scatter(logRho.iloc[-1], logT.iloc[-1], color='green', marker='o', s=100, label='Center')
    
    # Label the axes
    plt.ylabel(r'Log$(T)$ (K)', fontsize=12)
    plt.xlabel(r'Log$(\rho)$ (g/cm$^3$)', fontsize=12)
    plt.title('Density vs Temperature Profile', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save or show the plot
    plt.savefig('rho_t_profile.png', dpi=300)
    plt.show()


def plot_logP_vs_logR(df,
                      outfile:str=None,
                      colors:str=['purple','skyblue'], 
                      polytropes: list=None,
                      model_name:float=6, 
                      fac_vals=[4,4,4],
                      ssize=30,
                      model_fit_params:list=None,
                      x_lims:list=None,
                      y_lims:list=None,
                      xlbl:str=r'Log$(R) R_{\odot}$',
                      plot_linear_R:bool=False,
                      ttl:str='Pressure vs Radius Profile',
                      ):
    """
    Plots the logP vs logR profile of a star.

    Parameters:
        df (pd.DataFrame): DataFrame containing the profile data.
        polytropes (list): List of polytropes to plot.
        model_name (float): The model name of the star.
        fac_vals (list): List of factors to multiply the polytropic pressure by.
        ssize (int): Size of the scatter points.
    """
    # Extract the necessary columns
    logP = df['logP']

    if plot_linear_R:
        logR = 10**df['logR']

    else:
        logR = df['logR']

    # Create the plot
    plt.figure(figsize=(8, 6))

    plt.scatter(logR, logP, label='Pressure vs Radius',s=ssize,alpha=0.7)

    if polytropes:
        for n, fac, cls in zip(polytropes, fac_vals, colors):
            # Convert logRho to actual density
            rho = 10**df['logRho']
            
            # Compute polytropic pressure
            P = fac * rho**(1 + 1/n)  # fac acts as the polytropic constant K
            lp = np.log10(P)

            # Plot polytropic pressure vs radius
            plt.plot(logR, lp, label=fr'Polytrope, n = {n}', alpha=0.7, c=cls)
    
    if model_fit_params: 
        for model_fit in model_fit_params:
            if plot_linear_R:
                min_ag,max_ag = model_fit[0],model_fit[1]
                model_fit_mask = (10**df['logR'] > min_ag) & (10**df['logR'] < max_ag)
                df_fit = df[model_fit_mask]
                offset = model_fit[2]
                model_fit_color = model_fit[3]

                # Add shaded region
                plt.fill_between(10**df_fit['logR'], 
                                df_fit['logP'] - offset, 
                                df_fit['logP'] + offset, 
                                color=model_fit_color, alpha=0.3, label=model_fit[4])

            else:
                min_ag,max_ag = model_fit[0],model_fit[1]
                model_fit_mask = (df['logR'] > min_ag) & (df['logR'] < max_ag)
                df_fit = df[model_fit_mask]
                offset = model_fit[2]

                # Add shaded region
                plt.fill_between(df_fit['logR'], 
                                df_fit['logP'] - offset, 
                                df_fit['logP'] + offset, 
                                color=model_fit_color, alpha=0.3, label='Fit Region')
    # Label the axes  0.8
    plt.ylabel(r'Log$(P) dyn/cm^2$', fontsize=12)
    plt.xlabel(xlbl, fontsize=12)

    plt.title(ttl, fontsize=14)
    plt.legend(loc='lower left')

    plt.grid(True, linestyle='--', alpha=0.6)

    #plt.ylim(0, 10e2)
    if x_lims:
        plt.xlim(x_lims[0],x_lims[1])


    if y_lims:
        plt.ylim(y_lims[0],y_lims[1])
    #plt.yscale('log')
    #plt.xscale('log')

    if outfile:
        plt.savefig(outfile,dpi=400)


    plt.show()

def plotting_dlogT_dlogP_vs_R(df, 
                              color='blue', 
                              ssize=20, 
                              figsize=(8, 6), 
                              xlabel=r'Log$(R) R_{\odot}$', 
                              ylabel=r'dlogT/dlogP', 
                              title='dlogT/dlogP vs Radius Profile', 
                              outfile='dlogT_dlogP_vs_R.png',
                              convection_limit=[0.3,0.4]):
    """
    This function computes dlogT/dlogP and plots it vs R.
    
    Parameters:
    df (DataFrame): DataFrame containing the data with columns 'logP', 'logT', and 'logR'.
    color (str): Color of the scatter points.
    s (int): Size of the scatter points.
    figsize (tuple): Size of the figure.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    outfile (str): Name of the file to save the plot.
    """

    # Extract the necessary columns
    logP = df['logP']
    logT = df['logT']
    logR = 10**df['logR']

    # Compute dlogT/dlogP
    dlogT_dlogP = np.gradient(logT, logP)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.scatter(logR, dlogT_dlogP, color=color, s=ssize, label='dlogT/dlogP vs Radius')

    # Shade the region where dlogT/dlogP > convection limit
    plt.fill_between(logR, 
                     convection_limit[0], 
                     convection_limit[1], 
                     color='red', alpha=0.3, 
                     label=f'Convenction Range {convection_limit} ')

    # Label the axes
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or show the plot
    plt.savefig(outfile, dpi=300)
    plt.show()




def plot_abundances(df1, df2, label1, label2, convection_zone=[0.36, 0.4]):
    """
    Plots the abundances of different elements in a star as a function of radius.
    
    Parameters:
        df1 (pd.DataFrame): DataFrame containing the profile data.
        df2 (pd.DataFrame): DataFrame containing the profile data.
    """

    def plot_element(radius, abundance, label, color, linestyle='-', marker=None, alpha=0.7):
        plt.plot(radius, abundance, color=color, label=label, linestyle=linestyle, marker=marker, alpha=alpha, linewidth=3)

    # Extract the necessary columns
    radius1 = 10**df1['logR']
    radius2 = 10**df2['logR']

    elements = {
        'H': ('h1', 'lightblue', 'blue'),
        'He': ('he4', 'palevioletred', 'crimson'),
        'C': ('c12', 'lightgreen', 'green'),
        'N': ('n14', 'moccasin','orange')
    }

    # Use seaborn style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 8))

    for i, (element, (column, color1,color2)) in enumerate(elements.items()):
        plot_element(radius1, df1[column], f'{element} {label1}', color1, linestyle='--')
        plot_element(radius2, df2[column], f'{element} {label2}', color2, linestyle='-')


    plt.fill_betweenx([0, 1], convection_zone[0], convection_zone[1], color='gray', alpha=0.2, label='Convection zone')
    # Label the axes
    plt.xlabel(r'Radius ($R/R_{\odot}$)', fontsize=14)
    plt.yscale('log')
    plt.ylabel(r'mass fraction', fontsize=14)
    plt.title('Abundances vs Radius Profile', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or show the plot
    plt.savefig('abundances_profile.png', dpi=300, bbox_inches='tight')
    plt.show()



############################################################################################################
"""Gaia Stuff"""

import numpy as np
import pandas as pd


# Define the reference solar temperature (Teff_sun) in Kelvin
Teff_sun = 5777

def calculate_bolometric_correction(Teff):
    """
    Calculate BC_G based on the temperature range.
    
    Parameters:
    Teff : float or np.array
        Effective temperature in Kelvin.
        
    Returns:
    BC_G : float or np.array
        Bolometric correction.
    """
    # Coefficients for BC_G from Table 8.3
    coefficients_4000_8000K = [6.000e-2, 6.731e-5, -6.647e-8, 2.859e-11, -7.197e-15]
    coefficients_3300_4000K = [1.749e+0, 1.977e-3, 3.737e-7, -8.966e-11, -4.183e-14]

    BC_G = np.zeros_like(Teff, dtype=float)
    
    # Split the calculation based on the temperature range
    mask_4000_8000 = (Teff >= 4000) & (Teff <= 8000)
    mask_3300_4000 = (Teff >= 3300) & (Teff < 4000)
    
    # For 4000-8000 K range
    if np.any(mask_4000_8000):
        temp_diff = Teff[mask_4000_8000] - Teff_sun
        BC_G[mask_4000_8000] = sum(
            coeff * temp_diff**i for i, coeff in enumerate(coefficients_4000_8000K)
        )
    
    # For 3300-4000 K range
    if np.any(mask_3300_4000):
        temp_diff = Teff[mask_3300_4000] - Teff_sun
        BC_G[mask_3300_4000] = sum(
            coeff * temp_diff**i for i, coeff in enumerate(coefficients_3300_4000K)
        )
    
    return BC_G


def plot_bcg_vs_teff(df, 
                    color='cyan', 
                    ssize=20, 
                    figsize=(8, 6), 
                    xlabel=r'$T_{eff}$ [K]', 
                    ylabel=r'$BC_G$ [mag]', 
                    title='', 
                    outfile='bcg_plot.png',
                    x_lims=[8000,3000],
                    marker='o',
                    alpha=0.7,
                    edgecolor='k',
                    linewidth=0.5
                    ):
    """
    This function computes dlogT/dlogP and plots it vs R.
    
    Parameters:
    df (DataFrame): DataFrame containing the data with columns 'logP', 'logT', and 'logR'.
    color (str): Color of the scatter points.
    ssize (int): Size of the scatter points.
    figsize (tuple): Size of the figure.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    outfile (str): Name of the file to save the plot.
    x_lims (list): Limits for the x-axis.
    marker (str): Marker style for the scatter points.
    alpha (float): Transparency level of the scatter points.
    edgecolor (str): Edge color of the scatter points.
    linewidth (float): Line width of the scatter points' edges.
    """

    # Extract the necessary columns
    bcg = df['BC_G']
    t_eff = df['teff_gspphot']

    # Create the plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(t_eff, bcg, color=color, s=ssize, marker=marker, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label='Estimated Bolometric Corrections')

    # Invert x-axis
    plt.gca().invert_xaxis()  # Flip the x-axis for decreasing temperature

    # Label the axes
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    # Add title
    plt.title(title, fontsize=14)

    # Add legend
    plt.legend()

    # Set xlims
    plt.xlim(x_lims[0], x_lims[1])

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or show the plot
    plt.savefig(outfile, dpi=300)
    plt.show()



def list_data_columns(df):
    for i in df.columns:
        print(i)

def temperature_model(params, C, Fe_H):
    """
    Model for the inverse of effective temperature (5040 / T).
    """
    b0, b1, b2, b3, b4, b5 = params
    return b0 + b1 * C + b2 * C**2 + b3 * Fe_H + b4 * Fe_H**2 + b5 * Fe_H * C

def fit_function(C, Fe_H, b0, b1, b2, b3, b4, b5):
    return b0 + b1 * C + b2 * C**2 + b3 * Fe_H + b4 * Fe_H**2 + b5 * Fe_H * C



def calc_r_and_M_G(df,A_G=0):
    """
    Calculate the distance in parsecs and the absolute G-band magnitude.

    Parameters:
    df (pd.DataFrame): DataFrame containing the Gaia data.

    Returns:
    pd.DataFrame: DataFrame with the new columns 'r_pc' and
    'M_G' added.
    """

    # Calculate r in pc
    df['r_pc'] = 1000 / df['parallax']

    # M_G = G + 5 - 5log10r - A_G
    df['M_G'] = df['phot_g_mean_mag'] + 5 - 5 * np.log10(df['r_pc']) - A_G

    return df

def calc_luminosity_and_R(df):
    """
    Calculate the luminosity and radius of the star.

    Parameters:
    df (pd.DataFrame): DataFrame containing the Gaia data.

    Returns:
    pd.DataFrame: DataFrame with the new columns 'luminosity' and
    'radius' added.
    """

    # Calculate bolometric luminosity in solar luminosities
    M_bol_sun = 4.74
    df['bol_lum'] = 10**((df['M_G'] - df["BC_G"] - M_bol_sun) / -2.5)

    # Stefan-Boltzmann constant to cgs units
    sigma_sb = const.sigma_sb.cgs.value

    # Convert bolometric luminosity from L/L_sun to erg/s
    df['bol_lum_cgs'] = df['bol_lum'] * const.L_sun.cgs.value

    # Calculate radius in solar radii
    df['radius_emp'] = (np.sqrt(df['bol_lum_cgs']/(4*np.pi*sigma_sb*(df['T_empirical'])**4)))/const.R_sun.cgs.value

    return df



def plot_hr_diagram(data, output_file=None,):
    """
    Plots the Hertzsprung-Russell diagram using MESA's history data.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing MESA's history data.
        output_file (str): Optional. Path to save the plot as an image.
    """
    # Extract necessary columns: log_Teff and log_L
    log_Teff = data['T_empirical']
    log_L = data['bol_lum']

    # center_h1 as the color
    center_h1 = data['radius_emp']


    radii_lines  = [0.03,0.1,0.3,1,3,10,30,100]
    radii_labels = ['0.03','0.1','0.3','1','3','10',]
    radii_colors = ['purple','blue','green','orange','red','gray']
    
    
    scatter = plt.scatter(log_Teff, log_L, c=center_h1, cmap='viridis', edgecolor='k', s=50, alpha=0.7, label='Gaia Stars within 70 pc', norm=LogNorm())
    plt.gca().invert_xaxis()  # Flip the x-axis for decreasing temperature
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(r'Radius $R_{\odot}$', fontsize=12)

    plt.xscale('log')
    plt.yscale('log')

    # r_sun = r'$R_{\odot}$'
    # r_tol = 0.1

    # # Add the radii lines
    # for r,lbl,cls in zip(radii_lines,radii_labels,radii_colors):
    #     r_mask = (center_h1 > r - r_tol) & (center_h1 < r + r_tol)
    #     plt.scatter(log_Teff[r_mask], log_L[r_mask], c=cls, alpha=0.2, label=f'R={lbl} {r_sun}')
    
    # Label the axes
    plt.xlabel(r'Log$(T_{eff})$ (K)', fontsize=14)
    plt.ylabel(r'Log Luminosity (log$\frac{L}{L_{\odot}}$)', fontsize=14)
    plt.title('Hertzsprung-Russell Diagram', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


# Make a color magnitude diagram
def plot_color_magnitude_diagram(r):

    # Extract necessary columns (g band magnitude and bp-rp color)
    abs_mag_col = 'M_G'
    color_col = 'bp_rp'

    # Create a color map based on the BP-RP color index
    colors = r['radius_emp']

    # Create the plot
    fig, ax = plt.subplots(figsize=(16/2., 9/2.))
    scatter = ax.scatter(
        r[color_col],
        r[abs_mag_col], 
        s=20, 
        c=colors, 
        alpha=0.4,  # Increase transparency
        edgecolor='k',  # Optional for clarity
        label='Cluster Stars',
        cmap='plasma',
        norm=LogNorm()  # Apply logarithmic normalization
    )
    ax.set_xlabel('BP - RP [mag]', fontsize=12)
    ax.set_ylabel('Absolute Magnitude G band [mag]', fontsize=12)
    ax.set_title('Color-Magnitude Diagram', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add a color bar to show the colormap scale
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'Empirical Radius $R_{\odot}$', fontsize=12)


    ax.legend()

    plt.savefig('color_magnitude_diagram.png', dpi=400, bbox_inches='tight')

    plt.show()
