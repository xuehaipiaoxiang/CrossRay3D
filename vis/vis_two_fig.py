import matplotlib.pyplot as plt  
import numpy as np

# Data for the plot
token_sampling_ratio = [0.25, 0.50, 0.75, 1.00]
map_distribution_injection = [68.1, 70.0, 70.0, 70.0]
map_without_distribution_injection = [44.5, 57.6, 65.7, 69.3]

nds_distribution_injection = [70.9, 71.9, 72.3, 72.4]
nds_without_distribution_injection = [46.7, 59.5, 67.3, 70.1]

# Define the width of the bars
bar_width = 0.03

# Create a 1x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Set background color for the entire figure and axes
axs[0].set_facecolor('whitesmoke')    # Set the background color of the first subplot
axs[1].set_facecolor('whitesmoke')    # Set the background color of the second subplot

# First subplot
# Plot the line graphs for both datasets with labels
axs[0].plot(token_sampling_ratio, map_distribution_injection, 'o-', color='tab:orange', markersize=6, linewidth=2, label='w d.s.')
axs[0].plot(token_sampling_ratio, map_without_distribution_injection, 'o-', color='gray', markersize=6, linewidth=2, label='w/o d.s.')

# Set labels, title, and other properties for the first subplot
axs[0].set_xlabel('Keeping Token Ratio', fontsize=20)
axs[0].set_ylabel('mAP (%)', fontsize=20)
axs[0].set_xticks(token_sampling_ratio)
axs[0].set_yticks(range(30, 81, 10))
axs[0].grid(visible=True, axis='y', linestyle='--', linewidth=0.8, alpha=0.7)

# Add annotations for the data points
for x, y in zip(token_sampling_ratio, map_without_distribution_injection):
    if x == 0.50 or x == 0.75:
        axs[0].text(x , y - 4.0, f'{y:.1f}', color='gray', fontsize=16, ha='left', va='bottom')
        continue
    axs[0].text(x , y - 4.0, f'{y:.1f}', color='gray', fontsize=16, ha='center', va='bottom')

for x, y in zip(token_sampling_ratio, map_distribution_injection):
    axs[0].text(x, y + 0.6, f'{y:.1f}', color='tab:orange', fontsize=16, ha='center', va='bottom')

# Add legend to the first subplot with larger font size
axs[0].legend(fontsize=14, loc='lower right')  # Increase the font size here

# Second subplot
# Plot the line graphs for both datasets with labels
axs[1].plot(token_sampling_ratio, nds_distribution_injection, 'o-', color='tab:orange', markersize=6, linewidth=2, label='w d.s.')
axs[1].plot(token_sampling_ratio, nds_without_distribution_injection, 'o-', color='gray', markersize=6, linewidth=2, label='w/o d.s.')

# Set labels, title, and other properties for the second subplot
axs[1].set_xlabel('Keeping Token Ratio', fontsize=20)
axs[1].set_ylabel('NDS (%)', fontsize=20)
axs[1].set_xticks(token_sampling_ratio)
axs[1].set_yticks(range(30, 81, 10))
axs[1].grid(visible=True, axis='y', linestyle='--', linewidth=0.8, alpha=0.7)

axs[1].set_ylim(30, 80)
axs[1].set_xlim(0.2, 1.05)

# Add annotations for the data points with adjusted text positioning
for x, y in zip(token_sampling_ratio, nds_without_distribution_injection):
    if x == 0.50 or x == 0.75:
        axs[1].text(x, y - 4.0, f'{y:.1f}', color='gray', fontsize=16, ha='left', va='bottom')
        continue
    axs[1].text(x, y - 4.0, f'{y:.1f}', color='gray', fontsize=16, ha='center', va='bottom')
for x, y in zip(token_sampling_ratio, nds_distribution_injection):
    axs[1].text(x, y + 0.25, f'{y:.1f}', color='tab:orange', fontsize=16, ha='center', va='bottom')

# Add legend to the second subplot with larger font size
axs[1].legend(fontsize=14, loc='lower right')  # Increase the font size here

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot with high resolution
plt.savefig('./vis/fig_combined_identical_no_loop_adjusted_text.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
