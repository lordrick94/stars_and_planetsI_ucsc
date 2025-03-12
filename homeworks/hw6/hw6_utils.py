from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d

def query_stars(load=False):
    """
    Query Gaia DR3 for stars within a specified radius of given coordinates,
    filtering for stars with proper motion greater than pm_thresh.

    Parameters:

    Returns:
    - Pandas DataFrame containing the Gaia stars meeting the criteria.
    """

    # Construct query to get distance to the object by getting parralax of the closest star
    if load:
        df = pd.read_csv('d_query.csv')
    else:
        distance_query = """
        SELECT *
        FROM gaiadr3.gaia_source
        WHERE   parallax_over_error > 20
            AND (1000/parallax) < 100
        """
        print("Querying Gaia Archive for distance... This may take a few seconds.")
        # Row limit -1
        Gaia.ROW_LIMIT = -1
        dist_job = Gaia.launch_job_async(distance_query)
        dist_results = dist_job.get_results()
        df = dist_results.to_pandas()

        # Save the df
        df.to_csv('d_query.csv')

    return df

df_gaia = query_stars(load=True)

def count_stars_above_line(data, line_segments, u_abs = False):
    num_outliers = 0
    outlier_indices = []

    if u_abs:
        abs_mag = 'U_abs'
        color_mag = 'u-g'
    else:
        abs_mag = 'G_abs'
        color_mag = 'bp_rp'

    for idx, row in data.iterrows():
        x_star, y_star = row[color_mag], row[abs_mag]

        # Find the corresponding line segment
        for (x1, y1), (x2, y2) in line_segments:
            if x1 <= x_star <= x2:  # Check if star is within segment range
                # Linear interpolation of y on the line segment
                y_line = y1 + (y2 - y1) * (x_star - x1) / (x2 - x1)

                # Check if star is above the line (greater magnitude)
                if y_star > y_line:
                    num_outliers += 1
                    outlier_indices.append(idx)

                break  # Each star can be checked in one segment at most

    result_dict = {
        'num_outliers': num_outliers,
        'outlier_indices': outlier_indices,
    }

    return result_dict


# Make a color magnitude diagram
def plot_color_magnitude_diagram(r, 
                                 line_segments=None, 
                                 ssize=5,
                                 boring_plot=False,
                                 ot_line_color='cyan',
                                    mag_ug=False,
                                    save_file=None
                                 ):
    
    wd_sample = None

    if mag_ug:
        # Extract necessary columns (g band magnitude and u-g color)
        abs_mag_col = 'U_abs'
        color_col = 'u-g'
        xlbl = 'u - g [mag]'
        ylbl = 'Absolute Magnitude U band [mag]'
    else:
        # Extract necessary columns (g band magnitude and bp-rp color)
        abs_mag_col = 'G_abs'
        color_col = 'bp_rp'
        xlbl = 'BP - RP [mag]'
        ylbl = 'Absolute Magnitude G band [mag]'

    # Create the plot
    fig, ax = plt.subplots(figsize=(16/2., 9/2.))

    if line_segments:
        hw_part = 'c'
        result_dict = count_stars_above_line(r, line_segments,u_abs=mag_ug)

    
        # Plot the outliers
        ot_indices = result_dict['outlier_indices']
        mask = r.index.isin(ot_indices)
        ax.scatter(
            r[color_col][mask],
            r[abs_mag_col][mask],
            s=ssize,
            c=ot_line_color,
            alpha=0.4,
            edgecolor='k',
            label='White Dwarfs'
        )

        # Plot the rest of the stars
        ax.scatter(
            r[color_col][~mask],
            r[abs_mag_col][~mask],
            s=ssize,
            c='black',
            alpha=0.4,
            edgecolor='k',
            label='Other Stars'
        )

        print(f'Number of outliers: {result_dict["num_outliers"]}')
        wd_sample = r[mask]


        for segment in line_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2],c=ot_line_color, linestyle='--', linewidth=1,
                    )
            
    # Add annotation for the number of outliers line

        chose_pnt = line_segments[len(line_segments)//2][0]
        num_outliers = result_dict['num_outliers']
        ot_text = f'White Dwarfs: {num_outliers} stars'
        
        ax.annotate(ot_text,
             c=ot_line_color,
             xy=(chose_pnt[0], chose_pnt[1]),
             xytext=(chose_pnt[0] + 0.5, chose_pnt[1] - 1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color=ot_line_color),
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='grey', edgecolor=ot_line_color, alpha=0.5)
             )
        
        
    if boring_plot:
        # Create a color map based on the BP-RP color index
        colors = r['parallax_error']
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
        
    ax.set_xlabel(xlbl, fontsize=12)
    ax.set_ylabel(ylbl, fontsize=12)
    ax.set_title('Color-Magnitude Diagram', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)


    ax.legend()

    if save_file:
        plt.savefig(save_file, dpi=400, bbox_inches='tight')

    else:
        plt.savefig(f'cmd_part_{hw_part}.png', dpi=400, bbox_inches='tight')

    plt.show()


    return wd_sample

def query_sdss_xmatch(load=False):
    """
    Query Gaia DR3 for stars within a specified radius of given coordinates,
    filtering for stars with proper motion greater than pm_thresh.

    Parameters:

    Returns:
    - Pandas DataFrame containing the Gaia stars meeting the criteria.
    """
    if load:
        df = pd.read_csv('sdss_df.csv')
    else:
        distance_query = """
        SELECT g.source_id,g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
                g.parallax,g.parallax_error, q_sdss.objid, q_sdss.u, q_sdss.g, q_sdss.r, q_sdss.i, q_sdss.z


        FROM gaiadr3.sdssdr13_best_neighbour as best_neighbour

        INNER JOIN gaiadr3.gaia_source as g ON g.source_id = best_neighbour.source_id

        INNER JOIN external.sdssdr13_photoprimary as q_sdss ON q_sdss.objid = best_neighbour.original_ext_source_id

        WHERE (g.parallax > 10) 
        AND (g.phot_bp_mean_flux_over_error > 5) 
        AND (g.phot_rp_mean_flux_over_error > 5) 
        AND (g.parallax_over_error > 20)
        """
        print("Querying Gaia Archive for distance... This may take a few seconds.")
        # Row limit -1
        Gaia.ROW_LIMIT = -1
        dist_job = Gaia.launch_job_async(distance_query)
        dist_results = dist_job.get_results()
        df = dist_results.to_pandas()

        # Save the df
        df.to_csv('sdss_df.csv', index=False)

    return df




def plot_wd_cmd(df_wd,df_spec=None,mask_list=None,u_abs=False,
                clrs = ['crimson','lime','cyan'],lbls=['DA','DB','DQ'],
                ssize=5,marker='.',save_file=None,models=None,
                time_gridding=True,
                num_grids=6,
                model_plot_params:dict={'ms':5,
                                        'lw':1,
                                        'mc':['blue','red','green'],
                                        'mstyles':['-','--'],
                                        'mt_labels':['h2','he2']},):
    """
    Plot the color magnitude diagram of white dwarfs with the spectral types highlighted."

    Parameters:
    - df_wd: Pandas DataFrame containing the white dwarf sample.
    - df_spec: Pandas DataFrame containing the white dwarf spectral types.
    - mask_list: List of boolean masks for the spectral types.
    - clrs: List of colors for the spectral types.
    - lbls: List of labels for the spectral types.
    - ssize: Size of the markers.
    - marker: Marker style.
    - save_file: File name to save the plot.
    """
    fig, ax = plt.subplots(figsize=(9/1., 16/1.))

    model_type_labels = model_plot_params['mt_labels']
    model_linestyles = model_plot_params['mstyles']
    model_colors = model_plot_params['mc']

    if u_abs:
        # Extract necessary columns (g band magnitude and u-g color)
        abs_mag_col = 'U_abs'
        color_col = 'u-g'
        xlbl = 'u - g [mag]'
        ylbl = 'Absolute Magnitude U band [mag]'
    else:   
        # Extract necessary columns (g band magnitude and bp-rp color)
        abs_mag_col = 'G_abs'
        color_col = 'bp_rp'
        xlbl = 'BP - RP [mag]'
        ylbl = 'Absolute Magnitude G band [mag]'

    # Plot the white dwarfs
    ax.scatter(df_wd[color_col],df_wd[abs_mag_col],s=ssize,label='White Dwarfs',c='black',alpha=0.3,edgecolor=None,marker=marker)

    if mask_list:
        for mask,clr,lbl in zip(mask_list,clrs,lbls):
            ax.scatter(df_spec[mask][color_col],df_spec[mask][abs_mag_col],s=ssize,label=lbl,c=clr,alpha=0.9,edgecolor=None,marker=marker)

    if models:
        for j,(name,model_df) in enumerate(models.items()):            
            for i,model in enumerate(model_df):
                # Plot the models
                ax.plot(model[color_col],model[abs_mag_col],
                        label=fr'{name}$M_\odot$ {model_type_labels[i]}',
                        c=model_colors[j],alpha=0.5,
                        linestyle=model_linestyles[i],
                        markersize=model_plot_params['ms'],
                        linewidth=model_plot_params['lw'])
        
        if time_gridding:
            # Get the grid points for 1 Gyr
            model_age_grid_points = get_1Gyr_grid_points(models,
                                                         num_grids=4,
                                                         targ_age=1e9)
            for i in range(num_grids):
                _, extracted_list = extract_ith_tuple(model_age_grid_points, i)
                x_vals = [x for x, y in extracted_list]
                y_vals = [y for x, y in extracted_list]
                
                # Plot the line
                ax.plot(x_vals, y_vals, c='C8', alpha=0.7, linestyle='-', linewidth=2)
                
                # Annotate the midpoint with age
                if len(x_vals) > 1:  # Ensure there are enough points to annotate
                    mid_idx = len(x_vals) // 2  # Find the midpoint index
                    ax.text(x_vals[mid_idx], y_vals[mid_idx], f'{i+1} Gyr', 
                            fontsize=10, ha='center', va='bottom', color='black', 
                            bbox=dict(facecolor='white', edgecolor='black', 
                                      boxstyle='round,pad=0.2',alpha=0.6))

    ax.set_xlabel('BP - RP [mag]',fontsize=12)
    ax.set_ylabel('Absolute Magnitude G band [mag]',fontsize=12)
    ax.set_title('White Dwarf Color Magnitude Diagram',fontsize=14)
    ax.set_xlim(-0.5,2.)
    ax.set_ylim(5,20)
    ax.invert_yaxis()
    ax.grid(True,linestyle='--',alpha=0.6)
    ax.legend()

    if save_file:
        plt.savefig(save_file,dpi=800,bbox_inches='tight')

    else:
        plt.savefig('wd_cmd_masks.png',dpi=800,bbox_inches='tight')

    plt.show()


def get_mass_dfs(file:str): 
    # Define the file path
    file_path = f"/home/lordrick/Documents/Class_Repos/stars_and_planetsI_ucsc/homeworks/hw6/models/WD_CMD/Table_Mass_{file}"

    # Read file content
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Identify where tables start
    table_starts = [i for i, line in enumerate(lines) if line.strip().startswith("Teff")]  # Detect column headers

    # Ensure there are two tables detected
    if len(table_starts) < 2:
        raise ValueError("Could not detect two tables in the file.")

    # print(table_starts)

    # Read first plot_models(df_h2)table
    skiprows = 1
    table_end = table_starts[1] - 2
    df1 = pd.read_csv(file_path, sep=r'\s+', skiprows=skiprows, nrows=table_end - skiprows)

    # Read second table
    skiprows = table_starts[1] 
    df2 = pd.read_csv(file_path, sep=r'\s+', skiprows=skiprows)

    def get_bp_g(df_in):
        df = df_in.copy()
        df.loc[:,'bp_rp'] = df["G3_BP"] - df["G3_RP"]
        df.loc[:,'G_abs'] = df["G3"]
        return df

    df_h2 = get_bp_g(df1)

    df_he2 = get_bp_g(df2)

    return df_h2,df_he2


def get_z_new(df, z_new):
    """
    Interpolates the given data to find the x and y values at z_new.
    """
    # Example Data

    # Interpolation functions for x and y
    interp_x = interp1d(df['Age'], df['bp_rp'], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(df['Age'], df['G_abs'], kind='linear', fill_value="extrapolate")

    # Given z, find x and y
    x_new = float(interp_x(z_new))
    y_new = float(interp_y(z_new))

    return x_new, y_new


def get_1Gyr_grid_points(models,num_grids=4,targ_age=1e9):
    model_age_grid_points  = {}
    model_type_labels = ['h2','he2']
    for name,model in models.items():
        grid_points = {}
        for i,model_df in enumerate(model):
            # Find the index of the closest Age to 1 Gyr
            closest_index = (model_df.Age - targ_age).abs().idxmin()
            # Get the closest Age value
            closest_age = model_df.loc[closest_index, 'Age']
            z_new_vals = np.arange(closest_age,closest_age+(num_grids+1)*targ_age,targ_age)        
            grid_points[model_type_labels[i]] = [get_z_new(model_df,z_new) for z_new in z_new_vals]
        model_age_grid_points[name] = grid_points

    return model_age_grid_points

def extract_ith_tuple(model_age_grid_points, i):
    extracted_values = {}
    extracted_list = []

    for model_name, grid_points in model_age_grid_points.items():
        extracted_values[model_name] = {}

        for model_type, tuples_list in grid_points.items():
            if i < len(tuples_list):  # Ensure index is within bounds
                extracted_values[model_name][model_type] = tuples_list[i]
                extracted_list.append(tuples_list[i])
            else:
                extracted_values[model_name][model_type] = None  # Handle out-of-bounds case

    return extracted_values, extracted_list



