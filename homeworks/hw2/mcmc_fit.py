import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import corner

# Define the model function
def temperature_model(params, C, Fe_H):
    """
    Model for the inverse of effective temperature (5040 / T).
    """
    b0, b1, b2, b3, b4, b5 = params
    return b0 + b1 * C + b2 * C**2 + b3 * Fe_H + b4 * Fe_H**2 + b5 * Fe_H * C

# Define the log-likelihood
def log_likelihood(params, C, Fe_H, y, yerr):
    """
    Log-likelihood function for the MCMC fit.
    """
    model = temperature_model(params, C, Fe_H)
    return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr**2))

# Define the log-prior
def log_prior(params):
    """
    Log-prior for the parameters. Assume uniform priors within a reasonable range.
    """
    b0, b1, b2, b3, b4, b5 = params
    prior_lim = 2  # Prior limits for the parameters
    if -prior_lim < b0 < prior_lim and -prior_lim < b1 < prior_lim and -prior_lim < b2 < prior_lim and -prior_lim < b3 < prior_lim and -prior_lim < b4 < prior_lim and -prior_lim < b5 < prior_lim:
        return 0.0  # Log-prior is zero for uniform priors
    return -np.inf  # Log-prior is -inf for out-of-bound parameters

# Define the log-posterior
def log_posterior(params, C, Fe_H, y, yerr):
    """
    Log-posterior is the sum of log-prior and log-likelihood.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, C, Fe_H, y, yerr)

# Load your Gaia dataset
gaia_data = pd.read_csv('gaia_hw2_data.csv')

def run_mcmc_fit(gaia_data,
                 ndim=6,
                 nwalkers=32,
                 burn_in=50,
                 initial_guess=[1, 1, 1, 1, 1, 1],
                 nsteps=5000,
                 plot_mcmc_corner_plot=False,
                 plot_fit=False):
    """
    Run the MCMC fit on the Gaia data.
    """
    nan_mask = gaia_data["mh_gspphot"].isna()

    mask = (gaia_data["bp_rp"] > 0.33) & (gaia_data["bp_rp"] < 1.6)
    mask &= ~nan_mask 
    # Create a mask for NaN values in the 'mh_gspphot' column

    filtered_data  = gaia_data[mask]


    # Extract columns
    C = filtered_data["bp_rp"].values  # Color index
    Fe_H = filtered_data["mh_gspphot"].values  # Metallicity
    T = filtered_data["teff_gspphot"].values  # Effective temperature

    # Inverse temperature for the fit
    y = 5040 / T
    yerr = 0.05 * y  # Assume 5% uncertainty in y (adjust as appropriate)

    # Set up the MCMC sampler
    ndim = ndim  # Number of parameters (b0, b1, b2, b3, b4, b5)
    nwalkers = nwalkers  # Number of walkers
    initial_guess = initial_guess # Initial parameter guess
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(C, Fe_H, y, yerr)
    )

    # Run the MCMC chain
    print("Running MCMC...")
    nsteps = nsteps
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Analyze the results
    samples = sampler.get_chain(discard=burn_in, thin=10, flat=True)  # Flattened samples

    # Get parameter estimates (median and 1-sigma confidence intervals)
    param_names = ["b0", "b1", "b2", "b3", "b4", "b5"]
    results = {}
    for i, name in enumerate(param_names):
        median = np.percentile(samples[:, i], 50)
        lower = np.percentile(samples[:, i], 16)
        upper = np.percentile(samples[:, i], 84)
        results[name] = (median, lower, upper)

    # Print the results
    print("\nFitted parameters (median ± 1-sigma):")
    for name, (median, lower, upper) in results.items():
        print(f"{name} = {median:.4f} (+{upper-median:.4f}, -{median-lower:.4f})")

    if plot_mcmc_corner_plot:
        # Plot the corner plot
        fig = corner.corner(
            samples,
            labels=param_names,
            truths=[results[name][0] for name in param_names],
            show_titles=True,
        )
        plt.show()


    if plot_fit:
        # Plot the fit
        C_fit = np.linspace(0.33, 1.6, 500)
        T_fit = 5040 / temperature_model(
            [results[name][0] for name in param_names], C_fit, np.mean(Fe_H)
        )
        plot_T_vs_BP_RP(filtered_data, C_fit, T_fit)



    return results

def plot_T_vs_BP_RP(data, C_fit,results,param_names, 
                    lw=2,
                    ssize=20,
                    savefile=None):
    """
    Plot the effective temperature vs. BP-RP color.
    
    Parameters:
    df (DataFrame): DataFrame containing the data with columns 'bp_rp', 'teff_gspphot', and 'mh_gspphot'.
    C_fit (array-like): Array of BP-RP values for the best fit line.
    T_fit (array-like): Array of temperature values for the best fit line.
    savefile (str, optional): File path to save the plot. If None, the plot will be shown.
    """
    plt.figure(figsize=(12, 8))

    # Group the data by metallicity
    mask1 = data["mh_gspphot"] > -0.5
    mask2 = (data["mh_gspphot"] > -1.5) & (data["mh_gspphot"] <= -0.5)
    mask3 = (data["mh_gspphot"] > -2.5) & (data["mh_gspphot"] <= -1.5)
    mask4 = data["mh_gspphot"] <= -2.5

    grp_colors = ['salmon', 'midnightblue', 'mediumspringgreen', 'fuchsia']
    #fit_colors = ['orangered', 'lavender', 'limegreen', 'orchid']

    df = data.copy()
    # Assign groups based on metallicity
    df.loc[:, 'group'] = np.select([mask1, mask2, mask3, mask4], grp_colors)

    # Define labels for each group
    group_labels = {
        'salmon': '[Fe/H] > -0.5',
        'midnightblue': '-1.5 < [Fe/H] <= -0.5',
        'mediumspringgreen': '-2.5 < [Fe/H] <= -1.5',
        'fuchsia': '[Fe/H] <= -2.5'
    }

    median_metallicities = []

    # Plot the data
    for group in df['group'].unique():
        mask = df['group'] == group
        median_Fe_H = df[mask]['mh_gspphot'].median()
        median_metallicities.append(median_Fe_H)
        plt.scatter(df[mask]["bp_rp"], df[mask]["teff_gspphot"], c=group, label=group_labels[group], edgecolor="k", s=ssize, alpha=0.6, marker='o')

    # Plot the best fit line
        
    for i,Fe_H in enumerate(median_metallicities):
        
        T_fit = 5040 / temperature_model([results[name][0] for name in param_names], C_fit, Fe_H)
    
        plt.plot(C_fit, T_fit, color=grp_colors[i], label=f"[Fe/H] = {Fe_H:.3f}", lw=lw, linestyle="--", alpha=0.8)

    # Add labels and title
    plt.xlabel("BP-RP", fontsize=14)
    plt.ylabel("Effective Temperature (K)", fontsize=14)
    plt.title("Temperature vs BP-RP Color", fontsize=16)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save or show the plot
    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def calc_t_empirical(gaia_data, results, param_names):
    # Calculate the empirical temperature
    T_empirical = 5040 / temperature_model(
        [results[name][0] for name in param_names], gaia_data['bp_rp'], gaia_data["mh_gspphot"]
    )

    gaia_data["T_empirical"] = np.nan
    gaia_data.loc[:, "T_empirical"] = T_empirical

    return gaia_data