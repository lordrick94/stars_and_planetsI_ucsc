from hw3_utils import plot_cluster_parralax, plot_proper_motion
import os

def get_numeric_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def get_pmin_and_pmax_from_user():
    print("Enter the parralax limits for the cluster")
    p_min = get_numeric_input("Enter the minimum parralax value: ")
    p_max = get_numeric_input("Enter the maximum parralax value: ")

    print("Parralax limits: ", p_min, p_max)

    return p_min, p_max

def get_proper_motion_limits():
    print("Enter the proper motion limits for the cluster")
    bottom_left_x_coord = get_numeric_input("Enter the bottom left x coordinate: ")
    bottom_left_y_coord = get_numeric_input("Enter the bottom left y coordinate: ")
    width = get_numeric_input("Enter the width of the rectangle: ")
    height = get_numeric_input("Enter the height of the rectangle: ")

    bottom_left = (bottom_left_x_coord, bottom_left_y_coord)

    rec_spec = {
        "bottom_left": bottom_left,
        "width": width,
        "height": height,
    }
    print("Rectangle specification: ", rec_spec)


    return rec_spec 

def get_user_consent_to_continue(print_statement="Do you want to continue iterating? (y/n)"):
    it = 0
    while it == 0:
        user_input = input(f"{print_statement} ")
        
        if user_input == "y" or user_input == "Y":
            it = 1
            return True
        
        elif user_input == "n" or user_input == "N":
            it = 1
            return False
              
        else:
            print("Invalid input. Please enter 'y' or 'n'")


def run_iterations(r):
    # Remove all the saves plots
    cmd = "rm cluster_selection_plots/*.png"
    os.system(cmd)

    continue_iterating = True
    sq = 100
    i = 1

    while continue_iterating:
        print("Running iteration")

        repeat_step_1 = True
        while repeat_step_1:
            # Get the parralax limits from the user
            p_min, p_max = get_pmin_and_pmax_from_user()

            # Plot the parralax
            good_parralax = plot_cluster_parralax(r=r,
                                                bin_size=0.1,
                                                p_min=p_min,
                                                p_max=p_max,
                                                iteration_num=i,
                                                )
            repeat_step_1 = get_user_consent_to_continue("Do you want to re-enter the parralax limits? (y/n): ")
        
        repeat_step_2 = True
        while repeat_step_2:
            # Get the proper motion limits from the user
            rec_spec = get_proper_motion_limits()

            # Plot the proper motion
            good_proper_motion = plot_proper_motion(r=good_parralax,
                                                    rec_spec=rec_spec,
                                                    xlims=(-sq,sq),
                                                    ylims=(-sq ,sq),
                                                    iteration_num=i)
            
            repeat_step_2 = get_user_consent_to_continue("Do you want to re-enter the proper motion limits? (y/n): ")

        
        # Plot the cluster parralax again
        plot_cluster_parralax(r=good_proper_motion,
                              bin_size=0.1,
                              p_min=p_min,
                              p_max=p_max,
                                iteration_num='_final_'+str(i),
                              )
        
        # update the value of r
        r = good_proper_motion
        
        # Ask the user if they want to continue iterating
        continue_iterating = get_user_consent_to_continue("Do you want to continue iterating? (y/n): ")

        print("Continue iterating: ", continue_iterating)

        i += 1

    print("End of iterations")

    # Save the data to a file

    r.write("cluster_data_final.csv", format="csv", overwrite=True)

    print("Data saved to cluster_data_final.csv")

if __name__ == "__main__":
    # Load the data
    # Query the Gaia database for the cluster data
    from hw3_utils import gaia_query
    import astropy.units as u
    ra_hex = "08h 41m 48s"
    dec_hex = "+19d34m54s"
    radius = 5*u.deg
    cluster_data = gaia_query(ra_hex, dec_hex, radius,cons_scale=1,show=True,load_data=True)

    run_iterations(cluster_data)