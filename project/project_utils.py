from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
from bs4 import BeautifulSoup
import pandas as pd
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


def query_gaia_high_pm_stars(ra, dec, radius,distance):
    """
    Query Gaia DR3 for stars within a specified radius of given coordinates,
    filtering for stars with proper motion greater than pm_thresh.

    Parameters:
    - ra (float): Right Ascension (degrees)
    - dec (float): Declination (degrees)
    - radius (float): Search radius (arcminutes)
    - pm_thresh (float): Radial Velocity threshold (mas/yr)
    - max_rows (int): Maximum number of rows to return (default=10000)

    Returns:
    - Pandas DataFrame containing the Gaia stars meeting the criteria.
    """
    
    # Convert radius to degrees
    radius_deg = radius/60

    # Conver distance to parsecs
    frac = 0.3
    distance_pc = [(1000*distance*(1+frac)), (1000*distance*(1-frac))]

    print(f"Searching for stars within {radius} arcminutes of ({ra}, {dec})... between {distance_pc[0]} and {distance_pc[1]} parsecs")


    # Construct ADQL query to get the stars with proper motion > pm_thresh
    query = f"""
    SELECT source_id, ra, dec, parallax,parallax_error, pmra, pmdec, 
           pm, phot_g_mean_mag, radial_velocity
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), 
                     CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
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


def make_catalogue(df,load=False,max_retries=2):
    if load:
        df_new = pd.read_csv("gaia_snrs_catalogue.csv")
        q_out = {}
        for grp in df_new["Object"].unique():
            df_t = df_new[df_new["Object"]==grp]
            df_t = end_of_querry_processing(df_t)
            q_out[grp] = df_t
    else:
        q_out = {}

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
                    q["median_dist"] = df["median_dist"].values[i]
                    q["median_dist_error"] = df["median_dist_error"].values[i]
                    
                    q_out[Object_name] = q

                    if i == 0:
                        df_new = q
                    else: 
                        # Concatenate the new query results with the existing DataFrame
                        df_new = pd.concat([df_new, q], ignore_index=True)                          

                    df_new.to_csv("gaia_snrs_catalogue.csv", index=False)
                else:
                    print("query return empty df")
            except Exception as e:
                    print(f"Failed to query object {df['Object'].values[i]} at index {i}")
                    print(f"Error: {e}")  # Print the actual error message
                    continue
            
    df_new = end_of_querry_processing(df_new)

    return df_new,q_out 

def end_of_querry_processing(df_t):
    df_new = df_t.copy()
    df_new = df_new.reset_index(drop=True)
    # Convert parallax to pm from mas/yr to km/s
    df_new.loc[:,"dist_pc"] = 1000/df_new["parallax"]
    df_new.loc[:,"pm_km_s"] = df_new["pm"] * 4.74 * (1/df_new["parallax"])
    df_new.loc[:,"total_vel"] = (df_new["pm_km_s"]**2 + df_new["radial_velocity"]**2)**0.5

    return df_new


def plot_pm_vs_radial_velocity(df_new,median_dist=False,make_title=False): 
    # Make a scatter plot of the radial velocity vs. the proper motion, color-coded by object names
    plt.figure(figsize=(10, 6))

    sns.scatterplot(data=df_new, x='dist_pc', y='pm_km_s',hue='Object', palette='tab10', s=30,alpha=0.6)

    # Axvline at median distance
    if median_dist:
        # Plot a vertical fill between the median distance and the uncertainty
        dist = 1000*df_new["median_dist"].values[0]
        dist_error = 1000*df_new["median_dist_error"].values[0]
        plt.axvline(dist, color='r', linestyle='--', label=f"Median distance: {dist:.2f} pc")
        plt.fill_between([dist - dist_error, dist + dist_error], 0, df_new['pm_km_s'].max() + 10, color='r', alpha=0.1)

    plt.ylabel("Proper motion (km/s)", fontsize=18)
    plt.xlabel("Distance [pc]", fontsize=18)
    plt.grid(linestyle='--', alpha=0.5)
    if make_title:
        plt.title(df_new.Object[0], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
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


def plot_radec(r, plot_pm_dir=None,make_title=False):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(r["ra"], r["dec"], s=2, alpha=0.5, color='y') 

    if plot_pm_dir:
        mask = r["pm_km_s"] > plot_pm_dir
        pm_ra_deg = r["pmra"][mask]
        pm_dec_deg = r["pmdec"][mask]
        ax1.quiver(
            r["ra"][mask], r["dec"][mask], 
            pm_ra_deg, pm_dec_deg, 
            angles="xy", color="red", alpha=0.7
        )

    #Plot the one with the highest proper motion
    max_idx = r["pm_km_s"].idxmax()
    ax1.quiver(
        r["ra"][max_idx], r["dec"][max_idx], 
        r["pmra"][max_idx], r["pmdec"][max_idx], 
        angles="xy", color="blue", alpha=0.7
    )
    # Add annotation for the highest proper motion
    ax1.annotate(
        f"Highest pm:{r['pm_km_s'][max_idx]:.2f} km/s", 
        (r["ra"][max_idx], r["dec"][max_idx]), 
        textcoords="offset points", 
        xytext=(5,5), 
        ha='center'
    )
    if make_title:
        ax1.set_title(r.Object[0], fontfamily="serif", fontsize=20)

    ax1.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18)
    ax1.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
    ax1.tick_params(axis='both', right=True, top=True, width=2, length=8, direction='in', which='both', labelsize=14) 
    ax1.tick_params(axis='x', which='minor', length=4, width=2, direction='in') 
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    plt.show()
