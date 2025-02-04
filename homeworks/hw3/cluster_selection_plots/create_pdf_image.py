import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Organizing images by category into columns
parallax_images = [
    "cluster_parralax_iteration_num1.png", "cluster_parralax_iteration_num2.png", 
    "cluster_parralax_iteration_num3.png", "cluster_parralax_iteration_num4.png"
]

proper_motion_images = [
    "cluster_proper_motion_iteration_num1.png", "cluster_proper_motion_iteration_num2.png", 
    "cluster_proper_motion_iteration_num3.png", "cluster_proper_motion_iteration_num4.png"
]

parallax_final_images = [
    "cluster_parralax_iteration_num_final_1.png", "cluster_parralax_iteration_num_final_2.png", 
    "cluster_parralax_iteration_num_final_3.png", "cluster_parralax_iteration_num_final_4.png"
]

# Create figure with 4 rows and 3 columns
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

# Loop through images and plot them in respective columns
for i in range(4):
    for j, img_list in enumerate([parallax_images, proper_motion_images, parallax_final_images]):
        img_path = img_list[i]
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[i, j].imshow(img)
            # axes[i, j].set_title(img_path, fontsize=8)
        axes[i, j].axis('off')  # Hide axes

# Adjust layout and save the figure
plt.tight_layout()
pdf_path = "cluster_iterations.pdf"
plt.savefig(pdf_path, format="pdf")
plt.close()

# Return the generated PDF
pdf_path
