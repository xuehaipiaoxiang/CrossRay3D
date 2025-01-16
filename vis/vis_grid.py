import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# plt.rcParams['font.family'] = 'Arial'  # 使用无衬线字体 Arial

# Custom colormap to match the reference
colors = ['#E9F2E3', '#A7D8B2', '#8ED0BE', '#45A3C8', '#105C9C']  # Light green to blue gradient
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

# Example data for the 4x4 matrix
matrix_data = np.array([[68.3, 70.0, 70.0, 70.0], 
                        [68.3, 70.0, 70.0, 70.0],
                        [68.2, 70.0, 70.0, 70.0],
                        [68.1, 69.8, 69.8, 70.0]])

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Create the heatmap
cax = ax.imshow(matrix_data, cmap=custom_cmap, interpolation='nearest', vmin=68, vmax=71)

# Add the vertical colorbar on the left side of the heatmap
cbar_v = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.044, pad=0.06, location='right')
cbar_v.set_label('mAP(%)', fontsize=20)

# Annotate the heatmap with values
for i in range(matrix_data.shape[0]):
    for j in range(matrix_data.shape[1]):
        ax.text(j, i, f'{matrix_data[i, j]:.1f}', ha='center', va='center', color='black', fontsize=14)

# Add axis labels and set custom ticks
ax.set_xticks(np.arange(4))  # Index positions of the grid
ax.set_yticks(np.arange(4))
ax.set_xticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=14)  # Custom tick labels for x-axis
ax.set_yticklabels(['1.0', '0.75', '0.50', '0.25'], fontsize=14)  # Custom tick labels for y-axis (reversed)
ax.set_xlabel('Keeping Image Token Ratio', fontsize=16)
ax.set_ylabel('Keeping LiDAR Token Ratio', fontsize=16)

# Add gridlines
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Remove spines for cleaner visualization
ax.spines[:].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('./vis/heatmap_with_vbar_on_left_fixed_ticks.png', dpi=300)
plt.show()
