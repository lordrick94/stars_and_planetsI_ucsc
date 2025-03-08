import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def extract_numerical_distance(distance_text):
    """ Extracts the first numerical distance value in kpc from a string. """
    if not distance_text:
        return None
    # Match numbers followed by "kpc"
    match = re.search(r"([\d\.]+)\s*kpc", distance_text)
    return float(match.group(1)) if match else None

def get_snr_distance(snr_name):
    url = f"https://www.mrao.cam.ac.uk/surveys/snrs/snrs.{snr_name}.html"
    response = requests.get(url)

    if response.status_code != 200:
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

def make_distance_table(snr_list):
    data = []
    for snr in snr_list:
        distance_info, distance_value = get_snr_distance(snr)
        data.append({"SNR Name": snr, "Distance Info": distance_info, "Distance (kpc)": distance_value})

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Define SNRs to check
    snr_list = ["G120.1+1.4", "G130.7+3.1", "G34.6-0.5", "G53.7-2.2"]  # Add more if needed

    # Create a DataFrame with the distance information
    df = make_distance_table(snr_list)
    print(df)
