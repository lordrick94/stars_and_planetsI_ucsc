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

def ms_fit_func(x, a, b,c,d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def fit_ms_func(data, set_min_mag=8, set_max_mag=18, ms_indices=None):
    df = data.copy()

    # Remove NaN rows
    df = df.dropna(subset=['bp_rp', 'phot_g_mean_mag'])

    if ms_indices:
        df = df.loc[ms_indices]

    df = df[(df['phot_g_mean_mag'] > set_min_mag) & (df['phot_g_mean_mag'] < set_max_mag)]
    x = df['bp_rp']
    y = df['phot_g_mean_mag']

    popt, _ = curve_fit(ms_fit_func, x, y)

    return popt


def count_stars_above_line(data, line_segments, ms_offset=0.5):
    num_outliers = 0
    num_ms_stars = 0
    ms_indices = []
    outlier_indices = []

    for idx, row in data.iterrows():
        x_star, y_star = row['bp_rp'], row['phot_g_mean_mag']

        # Find the corresponding line segment
        for (x1, y1), (x2, y2) in line_segments:
            y3, y4 = y1 + ms_offset, y2 + ms_offset  
            if x1 <= x_star <= x2:  # Check if star is within segment range
                # Linear interpolation of y on the line segment
                y_line = y1 + (y2 - y1) * (x_star - x1) / (x2 - x1)

                # Check if star is above the line (greater magnitude)
                if y_star < y_line:
                    num_outliers += 1
                    outlier_indices.append(idx)

                # Check if star is within the second line
                y_line2 = y3 + (y4 - y3) * (x_star - x1) / (x2 - x1)

                if y_star < y_line2 and y_star > y_line:
                    num_ms_stars += 1
                    ms_indices.append(idx)
                break  # Each star can be checked in one segment at most

    result_dict = {
        'num_outliers': num_outliers,
        'num_ms_stars': num_ms_stars,
        'outlier_indices': outlier_indices,
        'ms_indices': ms_indices
    }

    return result_dict

# Make a color magnitude diagram
def plot_color_magnitude_diagram(r, 
                                 line_segments=None, 
                                 ssize=5,
                                 ms_offset=0.5,
                                 boring_plot=False,
                                 ot_line_color='cyan',
                                 ms_line_color='crimson',
                                 plot_ms_curve_fit=None):

    

    # Extract necessary columns (g band magnitude and bp-rp color)
    abs_mag_col = 'phot_g_mean_mag'
    color_col = 'bp_rp'

    # Create a color map based on the BP-RP color index
    colors = r['parallax_error']

    

    # Create the plot
    fig, ax = plt.subplots(figsize=(16/2., 9/2.))

    if line_segments:
        hw_part = 'c'
        result_dict = count_stars_above_line(r, line_segments, ms_offset=ms_offset)
        # Plot the main sequence stars
        ms_indices = result_dict['ms_indices']
        ax.scatter(
            r[color_col].iloc[ms_indices],
            r[abs_mag_col].iloc[ms_indices],
            s=ssize,
            c=ms_line_color,
            alpha=0.4,
            edgecolor='k',
            label='Main Sequence Stars'
        )

        # Plot the outliers

        ot_indices = result_dict['outlier_indices']
        ax.scatter(
            r[color_col].iloc[ot_indices],
            r[abs_mag_col].iloc[ot_indices],
            s=ssize,
            c=ot_line_color,
            alpha=0.4,
            edgecolor='k',
            label='Outlier Stars'
        )

        # Plot the rest of the stars
        ax.scatter(
            r[color_col].drop(index=ms_indices).drop(index=ot_indices),
            r[abs_mag_col].drop(index=ms_indices).drop(index=ot_indices),
            s=ssize,
            c='black',
            alpha=0.4,
            edgecolor='k',
            label='Other Stars'
        )


        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2],c=ot_line_color, linestyle='--', linewidth=1,
                    )
            
        # Draw a second line right above the first one
        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1+ms_offset, y2+ms_offset],c=ms_line_color, linestyle='--', linewidth=1,
                    )
            
    # Add annotation for the number of outliers line

        chose_pnt = line_segments[len(line_segments)//2][0]
        num_outliers = result_dict['num_outliers']
        num_ms_stars = result_dict['num_ms_stars']
        ot_text = f'Outlier Line: {num_outliers} stars'
        ms_text = f'Main Sequence: {num_ms_stars} stars'
        
        ax.annotate(ot_text,
             c=ot_line_color,
             xy=(chose_pnt[0], chose_pnt[1]),
             xytext=(chose_pnt[0] + 0.5, chose_pnt[1] - 1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color=ot_line_color),
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='grey', edgecolor=ot_line_color, alpha=0.5)
             )
        
        ax.annotate(ms_text,
             c=ms_line_color,
             xy=(chose_pnt[0], chose_pnt[1]+ms_offset),
             xytext=(chose_pnt[0] - 1.5, chose_pnt[1] + 2.5),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color=ms_line_color),
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='grey', edgecolor=ms_line_color, alpha=0.5)
             )
        

    if plot_ms_curve_fit:
        hw_part = 'e'
        # Fit the main sequence
        ms_popt = fit_ms_func(r,
                              set_max_mag=plot_ms_curve_fit['max_mag'],
                              set_min_mag=plot_ms_curve_fit['min_mag'],
                              ms_indices=result_dict['ms_indices'])
        x = np.linspace(plot_ms_curve_fit['xmin'], plot_ms_curve_fit['xmax'], 100)
        y = ms_fit_func(x, *ms_popt)
        ax.plot(x, y, color='cyan', linestyle='--', linewidth=1, label='Main Sequence Fit')

    if boring_plot:
        hw_part = 'b'
        scatter = ax.scatter(
        r[color_col],
        r[abs_mag_col], 
        s=ssize, 
        c=colors, 
        alpha=0.4,  # Increase transparency
        edgecolor='k',  # Optional for clarity
        label='Cluster Stars',
        cmap='plasma',
        norm=LogNorm()  # Apply logarithmic normalization
         )
        # Add a color bar to show the colormap scale
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'Parallax Error', fontsize=12)
        
    ax.set_xlabel('BP - RP [mag]', fontsize=12)
    ax.set_ylabel('Absolute Magnitude G band [mag]', fontsize=12)
    ax.set_title('Color-Magnitude Diagram', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)


    ax.legend()

    plt.savefig(f'cmd_part_{hw_part}.png', dpi=400, bbox_inches='tight')

    plt.show()


def mag_differences_and_flux_ratios(data, ms_indices,ot_indices):
    """
    This function calculates the magnitude differences between the outlier stars and the ms line fit.
    
    Parameters:
        data (DataFrame): The data containing the stars
        ms_indices (list): The indices of the main sequence stars
        outlier_indices (list): The indices of the outlier stars

    Returns:
        DataFrame: The magnitude differences between the outlier stars and the ms line fit
    
    """
    # Fit the main sequence
    ms_popt = fit_ms_func(data, ms_indices=ms_indices)
    ms_fit = ms_fit_func(data['bp_rp'], *ms_popt)
    ms_fit = pd.Series(ms_fit, index=data.index)

    # Calculate the magnitude differences
    data = data.copy()
    
    data.loc[:,'ms_fit'] = ms_fit
    data.loc[:,'mag_diff'] = data['phot_g_mean_mag'] - data['ms_fit']
    

    # Calculate the flux ratio using the magnitude differences
    data.loc[:,'flux_ratio'] = 10 ** (data['mag_diff']  * -0.4)
    

    # Plot the histograms
    df = data.loc[ot_indices]
    plot_mag_diff_histogram(df)
    plot_flux_ratio_histogram(df)

    return data



def plot_mag_diff_histogram(df, bin_size=0.1):

    fig, ax = plt.subplots(figsize=(16/2., 9/2.))
    ax.hist(df['mag_diff'], bins=np.arange(-2, 2, bin_size), color='dodgerblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Magnitude Difference [mag]', fontsize=12)
    ax.set_ylabel('Number of Stars', fontsize=12)
    ax.set_title('Magnitude Difference Histogram', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('mag_diff_hist.png', dpi=400, bbox_inches='tight')
    plt.show()


def plot_flux_ratio_histogram(df,bin_size=0.1):

    fig, ax = plt.subplots(figsize=(16/2., 9/2.))
    ax.hist(df['flux_ratio'], bins=np.arange(0, 3, bin_size), color='dodgerblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Flux Ratio', fontsize=12)
    ax.set_ylabel('Number of Stars', fontsize=12)
    ax.set_title('Flux Ratio Histogram', fontsize=14)

    ax.set_xlim(1, 3)

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('flux_ratio_hist.png', dpi=400, bbox_inches='tight')
    plt.show()


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

def plot_cluster_position(r,symbol_size=2,good=None,iteration_num=0):
    """
    This function plots the parralax of the cluster stars.

    Parameters:
        ra_vals (list): List of right ascension values
        dec_vals (list): List of declination values
        symbol_size (int): Size of the symbols
    """
    #clear figure and ax
    plt.clf()

    if good is not None:
        fig, ax1 = plt.subplots(figsize=(16/2., 9/2.))
        ax1.scatter(r["ra"], r["dec"], s=1, alpha=0.5, color='black')
        ax1.scatter(r["ra"][good], r["dec"][good], s=symbol_size, alpha=1, color='dodgerblue')
        ax1.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18) #Alway label your axes with units!
        ax1.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
        ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) #Inward pointing ticks are so much easier to see where the *data* are.  Also, let's have ticks on all sides.
        ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in') #Let's not forget minor tickmarks.
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        plt.show()

        plt.clf()


        fig, ax1 = plt.subplots(figsize=(16/2., 9/2.))
        ax1.scatter(r["ra"][good], r["dec"][good], s=symbol_size, alpha=1, color='dodgerblue')
        ax1.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18) #Alway label your axes with units!
        ax1.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
        ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) #Inward pointing ticks are so much easier to see where the *data* are.  Also, let's have ticks on all sides.
        ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in') #Let's not forget minor tickmarks.
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())


    else:
        # Now, we can use the parallax to select stars consistent with the cluster and plot the coordinates again

        fig, ax1 = plt.subplots(figsize=(16/2., 9/2.))

        ax1.scatter(r["ra"], r["dec"], s=symbol_size, alpha=0.5, color='black')
        ax1.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18) #Alway label your axes with units!
        ax1.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
        ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) #Inward pointing ticks are so much easier to see where the *data* are.  Also, let's have ticks on all sides.
        ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in') #Let's not forget minor tickmarks.
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

    #Save the figure
    plt.savefig('cluster_selection_plots/cluster_position_'+str(iteration_num)+'.png', dpi=400, bbox_inches='tight')

    plt.show()


def plot_cluster_parralax(r,bin_size=0.01,p_min=0.96,p_max=1.31,good_ind=None,iteration_num=0):
    """
    This function plots the parralax of the cluster stars.

    Parameters:
        parralax_vals (list): List of parralax values
        bin_size (float): Size of the bins
        p_min (float): Minimum parralax value
        p_max (float): Maximum parralax value
        xlims (tuple): Limits of the x-axis
    """

    fig, ax1 = plt.subplots(figsize=(16/2., 9/2.)) #16x9 ratio is visually appealing, close to the Golden Ratio

    
    if good_ind is not None:
        ax1.hist(r["parallax"][good_ind], bins=np.arange(-10,25,bin_size), log=True, color="dodgerblue")

    else:
        ax1.hist(r["parallax"], bins=np.arange(-10,25,bin_size), log=True, color="dodgerblue")
    ax1.set_xlabel(r"Parallax (mas)", fontfamily="serif", fontsize=18) #Make sure you know your units!
    ax1.set_ylabel(r"Number of Stars", fontfamily="serif", fontsize=18)
    ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) #Inward pointing ticks are so much easier to see where the *data* are.  Also, let's have ticks on all sides.
    ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
    ax1.grid(which='both', linestyle='--', alpha=0.6)

    # Add a title
    ax1.set_title(f"Parallax Distribution_iteration_number_{iteration_num}", fontfamily="serif", fontsize=20)

    ax1.axvline(x=p_min, lw=3, ls=":", color="black")
    ax1.axvline(x=p_max, lw=3, ls=":", color="black")
    

    xlims = []
    xlims.append(p_min - 0.5)
    xlims.append(p_max + 0.5)

    # Set x limits
    ax1.set_xlim(xlims[0],xlims[1])

    # Save the plot
    output_file = f'cluster_selection_plots/cluster_parralax_iteration_num{iteration_num}.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    
    # Show the plot
    plt.show()

    # Get the index of the good parralax values
    good_parralax = (r["parallax"] > p_min) & (r["parallax"] < p_max)

    print("Number of stars with good parralax: ",len(r[good_parralax]), " out of ", len(r))

    return r[good_parralax]


# Plot the proper motion
def plot_proper_motion(r,rec_spec,
                       xlims=None,
                       ylims=None,
                       iteration_num=0):
    """
    This function plots the proper motion of the cluster stars.

    Parameters:
        pmra_vals (list): List of proper motion in RA values
        pmdec_vals (list): List of proper motion in Dec values
        xlims (tuple): Limits of the x-axis
        ylims (tuple): Limits of the y-axis
    """

    fig, ax = plt.subplots(figsize=(16/2., 9/2.))

    # Plot the proper motion
    ax.scatter(r["pmra"], r["pmdec"], s=2, alpha=0.5, color='black')
    ax.set_xlabel(r"Proper Motion RA (mas/yr)", fontsize=12)
    ax.set_ylabel(r"Proper Motion Dec (mas/yr)", fontsize=12)
    ax.set_title("Proper Motion", fontsize=14)

    if xlims is not None:
        ax.set_xlim(xlims[0],xlims[1])
    
    if ylims is not None:
        ax.set_ylim(ylims[0],ylims[1])

    # Make an rectangle around stars with same proper motion
    ax.add_patch(plt.Rectangle(rec_spec["bottom_left"],rec_spec["width"],rec_spec["height"],linewidth=3,edgecolor='r',
                               facecolor='none',linestyle='--',label='Stars with similar proper motion'))

    # Add a title
    ax.set_title(f"Proper Motion_iteration_number_{iteration_num}", fontfamily="serif", fontsize=20)

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    output_file = f'cluster_selection_plots/cluster_proper_motion_iteration_num{iteration_num}.png'
    plt.savefig(output_file, dpi=400, bbox_inches='tight')

    plt.show()

    # Get the index of the good proper motion values
    good_proper_motion =    (r["pmra"] > rec_spec["bottom_left"][0]) & (r["pmra"] < rec_spec["bottom_left"][0] + rec_spec["width"]) & (r["pmdec"] > rec_spec["bottom_left"][1]) & (r["pmdec"] < rec_spec["bottom_left"][1] + rec_spec["height"])

    print("Number of stars with good proper motion: ",len(r[good_proper_motion]), " out of ", len(r))

    return r[good_proper_motion]
