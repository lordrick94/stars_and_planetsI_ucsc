from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from requests.exceptions import Timeout
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def get_snr_distance(snr_name, retries=3, timeout=10):
    url = f"https://www.mrao.cam.ac.uk/surveys/snrs/snrs.{snr_name}.html"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                break
        except Timeout:
            print(f"Timeout occurred for {snr_name}. Retrying ({attempt + 1}/{retries})...")
            if attempt == retries - 1:
                print(f"Failed to retrieve {snr_name} after {retries} attempts due to timeout.")
                return None, None
        except requests.RequestException as e:
            print(f"Request failed for {snr_name}: {e}")
            return None, None
    else:
        print(f"Failed to retrieve {snr_name}: {response.status_code}")
        return None, None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the <b> tag that contains "Distance:"
    bold_tag = soup.find("b", string="Distance:")

    if not bold_tag:
        return None, None  # No distance info found

    # Get the parent tag (dt or dl) that contains the full sentence
    parent_tag = bold_tag.find_parent()
    
    if not parent_tag:
        return None, None

    # Extract all text, including from nested tags
    distance_info = " ".join(parent_tag.stripped_strings).replace("Distance:", "").strip()

    # Extract numerical distance from the sentence
    distance_value = extract_numerical_distance(distance_info)

    return distance_info, distance_value

def clean_snr_name(snr_name):
    """
    Cleans SNR names by:
    1. Removing 'G'
    2. Splitting at '+' or '-'
    3. Removing unnecessary zeros while keeping significant ones
    4. Reconstructing the name
    5. Adding 'G' back at the start
    """

    # Step 1: Remove 'G' from the beginning
    snr_name = snr_name.lstrip("G")

    # Step 2: Split at '+' or '-'
    parts = re.split(r"(\+|-)", snr_name)

    # Step 3: Remove unnecessary zeros
    def remove_trailing_zeros(num):
        if "." in num:  # Ensure decimal point remains when needed
            float_num = float(num)
            new_num = str(float_num)
        else:
            new_num = num.lstrip("0")
        return new_num

    first_part = remove_trailing_zeros(parts[0])
    second_part = remove_trailing_zeros(parts[2])

    # Step 4: Reconstruct the name
    cleaned_name = f"G{first_part}{parts[1]}{second_part}"

    return cleaned_name

def extract_numerical_distance(distance_text):
    """ Extracts the first numerical distance value in kpc from a string. """
    if not distance_text:
        return None
    # Match numbers followed by "kpc"
    match = re.search(r"([\d\.]+)\s*kpc", distance_text)
    return float(match.group(1)) if match else None

def make_distance_table(df,load=False):
    if load:
        df = pd.read_csv('objects.csv')

    else:        
        df['SNR_Name'] = df['Object'].apply(clean_snr_name)

        # Initialize empty columns for Distance_Info and Distance_kpc
        df['Distance_Info'] = None
        df['Distance_kpc'] = None

        # Process each row and update the file after each row
        for idx, row in df.iterrows():
            snr_name = row['SNR_Name']
            distance_info, distance_kpc = get_snr_distance(snr_name)
            df.at[idx, 'Distance_Info'] = distance_info
            df.at[idx, 'Distance_kpc'] = distance_kpc

            # Save the DataFrame to CSV after processing each row
            df.to_csv("snr_distances.csv", index=False)

    return df


def query_gaia_high_pm_stars(ra, dec, radius,distance,radius_frac=0.25):
    """
    Query Gaia DR3 for stars within a specified radius of given coordinates,
    filtering for stars with proper motion greater than pm_thresh.

    Parameters:
    - ra, dec: The coordinates of the center of the search region in degrees.
    - radius: The radius of the search region in arcminutes.
    - pm_thresh: The minimum proper motion in mas/yr.
    - radius_frac: The fraction of the radius to search within.

    Returns:
    - Pandas DataFrame containing the Gaia stars meeting the criteria.
    """
    
    # Convert radius to degrees
    radius_deg = radius/60


    # Conver distance to parsecs
    frac = 0.3
    
    distance_pc = [(1000*distance*(1+frac)), (1000*distance*(1-frac))]

    print(f"Searching for stars within {radius_frac*radius} arcminutes of ({ra}, {dec})... between {distance_pc[0]} and {distance_pc[1]} parsecs")


    # Construct ADQL query to get the stars with proper motion > pm_thresh
    query = f"""
    SELECT source_id, ra, dec, parallax,parallax_error, pmra, pmdec, 
           pm, phot_g_mean_mag,bp_rp, radial_velocity
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), 
                     CIRCLE('ICRS', {ra}, {dec}, {radius_frac*radius_deg}))
                    AND parallax_over_error > 20
                    AND phot_bp_mean_flux_over_error >=5
                    AND phot_rp_mean_flux_over_error >=5
                    AND 1000/parallax < {distance_pc[0]}
                    AND 1000/parallax > {distance_pc[1]}

    """
    print("Querying Gaia Archive... This may take a few seconds.")
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    
    # Convert to Pandas DataFrame
    df = results.to_pandas()
    
    return df


def make_catalogue(df,load=False):
    if load:
        if 'dist_pc' in df.columns:
            df_new = pd.read_csv("gaia_snrs_catalogue.csv")
        else:
            df_new = end_of_querry_processing(pd.read_csv("gaia_snrs_catalogue.csv"))
        q_out = {}
        for grp in df_new["Object"].unique():
            df_t = df_new[df_new["Object"]==grp]
            print(f"Initiating end of Query Processing for {grp}...")
            df_t = end_of_querry_processing(df_t,add_score=True)
            print(f"End of Query Processing Successfull for {grp}!!")
            q_out[grp] = df_t
    else:
        q_out = {}
        dfs = []

        for i in range(0, len(df)):
            print(f"Querying object {df['Object'].values[i]}")
            try:
                q = query_gaia_high_pm_stars(ra=df['RA'].values[i],
                                            dec=df['Dec'].values[i],
                                                radius=df['Ang_size'].values[i], 
                                                distance=df["median_dist"].values[i])
                
                
                #Append if q is not empty
                if len(q) > 0:
                    Object_name = df["Object"].values[i]
                    q["Object"] = Object_name
                    q["Ang_size"] = df['Ang_size'].values[i]
                    
                    q["median_dist"] = df["median_dist"].values[i]
                    q["median_dist_error"] = df["median_dist_error"].values[i]
                    
                    q["snr_ra"] = df['RA'].values[i]
                    q["snr_dec"] = df['Dec'].values[i]

                   

                    print("Initiating end of Query Processing...")

                    q = end_of_querry_processing(q,add_score=True)

                    print("End of Query Processing Successfull!!")
                    
                    q_out[Object_name] = q
                    dfs.append(q)

                    if i == 0:
                        df_new = q
                    else: 
                        # Concatenate the new query results with the existing DataFrame
                        df_new = pd.concat(dfs, ignore_index=True)                          

                    df_new.to_csv("gaia_snrs_catalogue.csv", index=False)
                else:
                    print("query return empty df")
            except Exception as e:
                    print(f"Failed to get table for object {df['Object'].values[i]} at index {i}")
                    print(f"Error: {e}")  # Print the actual error message
                    continue
            
    

    return df_new,q_out 

def end_of_querry_processing(df_t,add_score=False):
    df_new = df_t.copy()
    df_new = df_new.reset_index(drop=True)
    # Convert parallax to pm from mas/yr to km/s
    df_new.loc[:,"dist_pc"] = 1000/df_new["parallax"]
    df_new.loc[:,"pm_km_s"] = df_new["pm"] * 4.74 * (1/df_new["parallax"])
    df_new.loc[:,"total_vel"] = (df_new["pm_km_s"]**2 + df_new["radial_velocity"]**2)**0.5

    snr_ra = df_t.snr_ra.values[0]
    snr_dec = df_t.snr_dec.values[0]
    snr_dist = 1000*df_t.median_dist.values[0]

    df_new.loc[:,'snr_sep'] = precompute_separations(df_t,snr_ra=snr_ra,snr_dec=snr_dec)
    
    if add_score:
        # Calculate score
        df_new = compute_star_scores(df_new,snr_ra,snr_dec,snr_dist)


    return df_new

def precompute_separations(data, snr_ra, snr_dec):
    """
    Precomputes the angular separations of all stars from the SNR center.

    Parameters:
        data: Pandas DataFrame containing 'ra' and 'dec' columns.
        snr_ra: Right Ascension of SNR (degrees).
        snr_dec: Declination of SNR (degrees).

    Returns:
        A NumPy array with separations in arcseconds.
    """
    # Convert all star coordinates into SkyCoord object
    star_coords = SkyCoord(ra=data["ra"].values * u.deg, dec=data["dec"].values * u.deg, frame="icrs")

    # SNR coordinates
    snr_coords = SkyCoord(ra=snr_ra * u.deg, dec=snr_dec * u.deg, frame="icrs")

    # Compute separations in bulk (vectorized operation)
    separations = snr_coords.separation(star_coords).arcsecond  # Convert to arcseconds

    return separations

def plot_pm_vs_radial_velocity(df_new, median_dist=False, 
                               make_title=False, add_score=False,
                               savefile=None):
    # Make a scatter plot of the radial velocity vs. the proper motion, color-coded by object names
    fig, ax = plt.subplots(figsize=(10, 6))

    if add_score:
        scatter = ax.scatter(df_new["dist_pc"], df_new["pm_km_s"], alpha=0.5, c=df_new['score'], cmap='viridis')
        cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05,use_gridspec=False,location='top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Star score', fontsize=20)

        # Add highest score star
        max_idx = df_new["score"].idxmax()
        ax.scatter(df_new["dist_pc"][max_idx], 
               df_new["pm_km_s"][max_idx], s=300, c='red', marker='*', label='Top Candidate',alpha=0.7)

    else:
        scatter = sns.scatterplot(data=df_new, x='dist_pc', 
                                  y='pm_km_s', 
                                  hue='Object',
                                 palette='tab10', 
                                 s=30, alpha=0.6, ax=ax)

    # Axvline at median distance
    if median_dist:
        # Plot a vertical fill between the median distance and the uncertainty
        dist = 1000 * df_new["median_dist"].values[0]
        dist_error = 1000 * df_new["median_dist_error"].values[0]
        ax.axvline(dist, color='r', linestyle='--', label=f"Median distance: {dist:.2f} pc")
        ax.fill_between([dist - dist_error, dist + dist_error], 0, 
                        df_new['pm_km_s'].max() + 10, color='Grey', alpha=0.1,
                        label=f"Distance uncertainty: {dist_error:.2f} pc")

    
    ax.set_ylabel("Proper motion (km/s)", fontsize=18)
    ax.set_xlabel("Distance [pc]", fontsize=18)
    ax.grid(linestyle='--', alpha=0.5)
    if make_title:
        ax.set_title(df_new.Object[0], fontsize=20)
    ax.tick_params(axis='both', which='both', labelsize=18)

    # Add a legend
    ax.legend(fontsize=16, loc='upper right', framealpha=0.5)

    if savefile:
        plt.savefig(savefile,dpi=800,bbox_inches='tight')

    plt.show()

def split_value_uncertainty(value):
    if isinstance(value, str):
        if "±" in value:
            val, unc = value.split("±")
            return float(val), float(unc)
        elif "<" in value or ">" in value:
            val = float(value.replace("<", "").replace(">", ""))
            return val, 0.2 * val
        elif "–" in value:  # Handling ranges
            parts = value.split("–")
            val = (float(parts[0]) + float(parts[1])) / 2
            return val, 0.2 * val
        else:
            val = float(value)
            return val, 0.2 * val
    return value, None  # If it's not a string, return as is

def process_raw_distance_cat(df):
    df['test_d'] = df["Distance_kpc"].astype(str).apply(split_value_uncertainty)
    df['median_dist'],df['median_dist_error'] = zip(*df['test_d'])
    df = df.drop(columns=['test_d'])
    df.to_csv("dist_cat2.csv", index=False)
    return df


def plot_radec(df, plot_pm_dir=None,
               make_title=False,draw_star_ID=None,
               add_score=False,
               savefile=None):
    fig, ax1 = plt.subplots(figsize=(10,10))

    if add_score:
        scatter = ax1.scatter(df["ra"], df["dec"], s=20, alpha=0.5, c=df['score'])

        cbar = fig.colorbar(scatter,ax=ax1)
        cbar.set_label('Star score', fontsize=14) 

    else:
        ax1.scatter(df["ra"], df["dec"], s=5, alpha=0.5, color='y') 

    def make_quiver(df,mask,clr,annotate=False):
        pm_ra_deg = df["pmra"][mask]
        pm_dec_deg = df["pmdec"][mask]
        ax1.quiver(
            df["ra"][mask], df["dec"][mask], 
            pm_ra_deg, pm_dec_deg, 
            angles="xy", color=clr, alpha=0.7
        )

        if annotate:
            ax1.annotate(
                f"PM: {df['pm_km_s'][mask]:.2f} km/s",  # Text
                xy=(df["ra"][mask], df["dec"][mask]),  # Position of annotation
                textcoords="offset points", 
                xytext=(5, 15),  # Offset for text
                ha='center', 
                fontsize=18,
                color='crimson'  # Use 'color' instead of 'c'
            )

    if draw_star_ID:
        # Make a star given the star ID
        mask = df["source_id"] == draw_star_ID
        make_quiver(df,mask,"cyan",annotate=True)

    if plot_pm_dir:
        mask = df["score"] > plot_pm_dir
        make_quiver(df,mask,"blue",annotate=False)

    #Plot the one with the highest proper motion
    max_idx = df["score"].idxmax()
    ax1.quiver(
        df["ra"][max_idx], df["dec"][max_idx], 
        df["pmra"][max_idx], df["pmdec"][max_idx], 
        angles="xy", color="crimson", alpha=0.7,
        label= ""
    )

    ax1.scatter(df["ra"][max_idx], df["dec"][max_idx], s=150, c='red', marker='*', label='Top Candidate',alpha=0.7)


    ax1.annotate(
        f"PM: {df['pm_km_s'][max_idx]:.2f} km/s",  # Text
        xy=(df["ra"][max_idx], df["dec"][max_idx]),  # Position of annotation
        textcoords="offset points", 
        xytext=(5, 15),  # Offset for text
        ha='center', 
        fontsize=18,
        color='crimson'  # Use 'color' instead of 'c'
    )
    if make_title:
        ax1.set_title(df.Object[0], fontfamily="serif", fontsize=20)

    ax1.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18)
    ax1.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
    ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) 
    ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in') 
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Add a legend
    ax1.legend(loc='upper right',fontsize=18,framealpha=0.5)

    if savefile:
        plt.savefig(savefile,dpi=800,bbox_inches='tight')

    plt.show()


def compute_sep_score(snr_coord_sep, snr_diameter_arcmin):
    """
    Computes a scaled separation score based on the size of the SNR.
    - 3 points if within 10% of the SNR size.
    - 2 points if within 20%.
    - 1 point if within 25%.
    - Smoothly decreases beyond 25%.
    
    Parameters:
        row: DataFrame row containing star's RA & Dec.
        snr_ra: Right Ascension of the SNR center (degrees).
        snr_dec: Declination of the SNR center (degrees).
        snr_diameter_arcmin: Angular diameter of the SNR (arcseconds).

    Returns:
        Separation Score (0 to 3)
    """
    separation = snr_coord_sep

    # Define scaling thresholds based on SNR size
    d1 = 0.1 * snr_diameter_arcmin  # 10% of SNR size (score 3)
    d2 = 0.2 * snr_diameter_arcmin  # 20% of SNR size (score 2)
    d3 = 0.25 * snr_diameter_arcmin  # 25% of SNR size (score 1)

    # Compute scaled score with smooth transition
    if separation < d1:
        sep_score = 3  # Within 10% of SNR
    elif separation < d2:
        sep_score = 3 - (separation - d1) / (d2 - d1)  # Linearly interpolate from 3 → 2
    elif separation < d3:
        sep_score = 2 - (separation - d2) / (d3 - d2)  # Linearly interpolate from 2 → 1
    else:
        sep_score = max(0, 1 - (separation - d3) / (d3))  # Smoothly approaches 0

    return sep_score

def compute_fine_score(row, snr_ra, snr_dec, snr_distance):
    # Compute PM direction angle (continuous scaling)
    star_pm_vector = np.array([row["pmra"], row["pmdec"]])
    snr_vector = np.array([row["ra"] - snr_ra, row["dec"] - snr_dec])
    theta = np.degrees(np.arccos(np.dot(star_pm_vector, snr_vector) / 
                                 (np.linalg.norm(star_pm_vector) * np.linalg.norm(snr_vector) + 1e-6)))  # Avoid division by zero

    # Scale PM direction angle smoothly from 0 (worst) to 3 (best)
    pm_angle_score = max(0, (90 - theta) / 30)  # Closer to 0° gives a higher score, maxes at 3

    # Compute score for distance from SNR coords
    sep_score = compute_sep_score(row["snr_sep"],snr_diameter_arcmin=row["Ang_size"])  

    # Compute proper motion (PM) score
    pm = row['pm_km_s']
    pm_score = np.interp(pm, [0, 10, 50, 500, 1000], [0, 1, 2, 3, 2])  # Smooth transition

    # Compute color score 
    color_score = np.interp(row["bp_rp"], [-0.2, 0.5, 1.5, 2.5], [3, 2.5, 1.5, 1])  # Blue stars get the highest scores

    # Compute magnitude score 
    mag_score = np.interp(row["phot_g_mean_mag"], [10, 13, 15, 18], [1, 2, 2.5, 3])  # Fainter stars score higher

    # Compute distance score 
    star_distance = row["dist_pc"]
    distance_error = abs(star_distance - snr_distance) / snr_distance
    distance_score = np.interp(distance_error, [0, 0.2, 0.5, 1], [3, 2, 1, 0])  # Close distances get higher scores

    # Compute final score
    weigths = {
        "pm_angle": 3,
        "pm": 3,
        "mag": 2,
        "color": 4,
        "distance": 5,
        "separation":4,
    }
    final_score = (pm_angle_score * weigths["pm_angle"] +
                   pm_score * weigths["pm"] +
                   mag_score * weigths["mag"] +
                   color_score * weigths["color"] +
                   distance_score * weigths["distance"] +
                   sep_score* weigths["separation"])
    
    final_score = final_score/(3*sum(weigths.values())) * 100  # Normalize to 0-100 

    return final_score

def compute_star_scores(df, snr_ra, snr_dec, snr_distance):
    df["score"] = df.apply(compute_fine_score, args=(snr_ra, snr_dec, snr_distance), axis=1)

    df = df.sort_values(by='score',ascending=False)

    return df


def plot_cmd(df_full,ssize=30,marker='.',save_file=None):
    fig, ax = plt.subplots(figsize=(16./2, 9./2))

    # Plot the white dwarfs
    scatter = ax.scatter(df_full['bp_rp'],df_full['phot_g_mean_mag'],
               s=ssize,label='Star candidates',c=df_full['score'],
               alpha=0.7,edgecolor=None,marker=marker)
    
    # Add highes score star
    max_idx = df_full["score"].idxmax()

    ax.scatter(df_full['bp_rp'][max_idx],df_full['phot_g_mean_mag'][max_idx],
                s=ssize*10,label='Top Candidate',c='red',
                alpha=0.7,edgecolor='black',marker='*',)

    ax.set_xlabel('BP - RP [mag]',fontsize=12)
    ax.set_ylabel('Absolute Magnitude G band [mag]',fontsize=12)
    ax.set_title('SNR Binary Companion Color Magnitude Diagram',fontsize=14)
    ax.invert_yaxis()
    ax.grid(True,linestyle='--',alpha=0.6)

    # Add a color bar
    cbar = fig.colorbar(scatter,ax=ax)
    cbar.set_label('Star Score', fontsize=12)
    
    ax.legend(fontsize=16,loc='upper right')

    if save_file:
        plt.savefig(save_file,dpi=800,bbox_inches='tight')


