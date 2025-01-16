# All statistical data load from Stram PETR Backbone
# watch local variables like "weight[2,4000:5000].flatten()[cls_index[2,4000:5000].flatten()==5]""

car =  [0.8008, 0.7940, 0.7921, 0.7917, 0.7900, 0.7889, 0.7876, 0.7860, 0.7851,
        0.7847, 0.7837, 0.7822, 0.7818, 0.7814, 0.7810, 0.7802, 0.7788, 0.7787,
        0.7770, 0.7757, 0.7739, 0.7732, 0.7731, 0.7729, 0.7725, 0.7725, 0.7701,
        0.7685, 0.7676, 0.7674, 0.7662, 0.7661, 0.7660, 0.7656, 0.7656, 0.7655,
        0.7647, 0.7643, 0.7642, 0.7632, 0.7631, 0.7629, 0.7623, 0.7621, 0.7610,
        0.7597, 0.7596, 0.7596, 0.7596, 0.7591, 0.7579, 0.7579, 0.7577, 0.7577,
        0.7577, 0.7574, 0.7571, 0.7570, 0.7566, 0.7558, 0.7554, 0.7550, 0.7539,
        0.7536, 0.7536, 0.7525, 0.7522]
car2 = [0.7935, 0.7736, 0.7735, 0.7697, 0.7695, 0.7669, 0.7663, 0.7662, 0.7659,
        0.7655, 0.7634, 0.7632, 0.7631, 0.7610, 0.7609, 0.7605, 0.7587, 0.7587,
        0.7583, 0.7582, 0.7577, 0.7575, 0.7568, 0.7563, 0.7562, 0.7556, 0.7551,
        0.7546, 0.7535, 0.7531, 0.7514, 0.7512, 0.7508, 0.7503, 0.7502, 0.7500,
        0.7495, 0.7489, 0.7489, 0.7484, 0.7483, 0.7478, 0.7477, 0.7470] + [0.7949, 0.7805, 0.7804, 0.7769, 0.7767, 0.7744, 0.7732, 0.7728, 0.7726,
        0.7724, 0.7704, 0.7703, 0.7703, 0.7691, 0.7690, 0.7684, 0.7675, 0.7674,
        0.7668, 0.7667, 0.7657, 0.7657, 0.7651, 0.7644, 0.7642, 0.7635, 0.7629,
        0.7623, 0.7614, 0.7611, 0.7597, 0.7592, 0.7588, 0.7586, 0.7582, 0.7580,
        0.7576, 0.7571, 0.7571, 0.7563, 0.7562, 0.7559, 0.7558, 0.7548]


car = car + car2


truck = [0.8992, 0.8971, 0.8966, 0.8958, 0.8928, 0.8910, 0.8909, 0.8904, 0.8894,
        0.8889, 0.8889, 0.8882, 0.8876, 0.8870, 0.8869, 0.8869, 0.8856, 0.8855,
        0.8851, 0.8846, 0.8845, 0.8843, 0.8837, 0.8833, 0.8827, 0.8825, 0.8824,
        0.8821, 0.8809, 0.8807, 0.8801, 0.8799, 0.8796, 0.8795, 0.8795, 0.8793,
        0.8783, 0.8777, 0.8768, 0.8768, 0.8768, 0.8767, 0.8765, 0.8762, 0.8760,
        0.8758, 0.8755, 0.8750, 0.8745, 0.8744, 0.8740, 0.8739, 0.8737, 0.8730,
        0.8729, 0.8728, 0.8726, 0.8719, 0.8712, 0.8711, 0.8709, 0.8709, 0.8704,
        0.8704, 0.8698, 0.8696, 0.8693, 0.8691, 0.8691, 0.8690, 0.8689, 0.8686,
        0.8685, 0.8683, 0.8681, 0.8680, 0.8679, 0.8678, 0.8677, 0.8676, 0.8675,
        0.8672, 0.8662, 0.8656, 0.8649, 0.8649, 0.8649, 0.8634, 0.8632, 0.8619,
        0.8619, 0.8619, 0.8618, 0.8617, 0.8614, 0.8613, 0.8612, 0.8612, 0.8611,
        0.8610, 0.8608, 0.8604, 0.8602, 0.8601, 0.8598, 0.8594, 0.8592, 0.8590,
        0.8583, 0.8583, 0.8575, 0.8570, 0.8570, 0.8569, 0.8567, 0.8551, 0.8550,
        0.8548, 0.8547, 0.8546, 0.8544, 0.8540, 0.8540, 0.8540, 0.8537, 0.8536,
        0.8534, 0.8532, 0.8531, 0.8527, 0.8527, 0.8524, 0.8523, 0.8521, 0.8521,
        0.8520, 0.8519, 0.8518, 0.8516, 0.8512, 0.8511, 0.8507, 0.8504, 0.8498,
        0.8492, 0.8488, 0.8486, 0.8483, 0.8479, 0.8475, 0.8473, 0.8461, 0.8460,
        0.8454, 0.8453, 0.8451, 0.8449, 0.8446, 0.8440, 0.8439, 0.8438, 0.8437,
        0.8437, 0.8437, 0.8435, 0.8435, 0.8430, 0.8421, 0.8420, 0.8410, 0.8408,
        0.8399, 0.8394, 0.8393, 0.8387, 0.8387, 0.8381, 0.8380, 0.8379, 0.8378,
        0.8377, 0.8369, 0.8367, 0.8365, 0.8360, 0.8357, 0.8356, 0.8356, 0.8355,
        0.8354, 0.8354, 0.8349, 0.8348, 0.8348, 0.8347, 0.8347, 0.8346, 0.8338,
        0.8336, 0.8328, 0.8319, 0.8313, 0.8309, 0.8308, 0.8307, 0.8300, 0.8298,
        0.8295, 0.8291, 0.8286, 0.8284, 0.8284, 0.8284, 0.8281, 0.8274, 0.8271,
        0.8271, 0.8269, 0.8262, 0.8255, 0.8254, 0.8253, 0.8252, 0.8247, 0.8239,
        0.8238, 0.8238, 0.8234, 0.8231, 0.8230, 0.8226, 0.8225, 0.8224, 0.8222,
        0.8217, 0.8207, 0.8207, 0.8207, 0.8200, 0.8199, 0.8196, 0.8194, 0.8190,
        0.8190, 0.8190, 0.8188, 0.8186, 0.8182, 0.8175, 0.8170, 0.8169, 0.8165,
        0.8164, 0.8162, 0.8162, 0.8161, 0.8157, 0.8147, 0.8142, 0.8139, 0.8138,
        0.8129, 0.8128, 0.8126, 0.8125, 0.8124, 0.8120, 0.8120, 0.8120, 0.8117,
        0.8116, 0.8116, 0.8111, 0.8110, 0.8107, 0.8105, 0.8102, 0.8101, 0.8095,
        0.8092, 0.8087, 0.8083, 0.8082, 0.8077, 0.8075, 0.8072, 0.8071, 0.8071,
        0.8070, 0.8065, 0.8063, 0.8063, 0.8061, 0.8060, 0.8058, 0.8057, 0.8054,
        0.8052, 0.8051, 0.8051, 0.8046, 0.8045, 0.8045, 0.8042, 0.8038, 0.8037,
        0.8037, 0.8037, 0.8036, 0.8034, 0.8033, 0.8031, 0.8029, 0.8029, 0.8028,
        0.8028, 0.8023, 0.8021, 0.8017, 0.8014, 0.8013, 0.8010, 0.8010, 0.8008,
        0.8005, 0.8003, 0.8002, 0.8001, 0.7998, 0.7998, 0.7994, 0.7993, 0.7991,
        0.7990, 0.7988, 0.7986, 0.7983, 0.7971, 0.7970, 0.7966, 0.7966, 0.7966,
        0.7965, 0.7963, 0.7962, 0.7960, 0.7960, 0.7960, 0.7958, 0.7958, 0.7956,
        0.7953, 0.7953, 0.7952, 0.7951, 0.7951, 0.7943, 0.7942, 0.7941, 0.7936,
        0.7931, 0.7930, 0.7930, 0.7929, 0.7925, 0.7924, 0.7923, 0.7923, 0.7919,
        0.7916, 0.7914, 0.7911, 0.7910, 0.7909, 0.7909, 0.7908, 0.7906, 0.7904,
        0.7904, 0.7902, 0.7900, 0.7898, 0.7898, 0.7897, 0.7896, 0.7893, 0.7892,
        0.7889, 0.7886, 0.7880, 0.7880, 0.7879, 0.7871, 0.7867, 0.7866, 0.7860,
        0.7860, 0.7859, 0.7852, 0.7849, 0.7849, 0.7847, 0.7847, 0.7847, 0.7846,
        0.7846, 0.7845, 0.7843, 0.7843, 0.7838, 0.7837, 0.7837, 0.7836, 0.7833,
        0.7832, 0.7831, 0.7828, 0.7827, 0.7827, 0.7824, 0.7824, 0.7823, 0.7819,
        0.7817, 0.7814, 0.7812, 0.7810, 0.7807, 0.7806, 0.7801, 0.7798, 0.7796,
        0.7794, 0.7793, 0.7792, 0.7791, 0.7787, 0.7786, 0.7786, 0.7785, 0.7784,
        0.7784, 0.7784, 0.7784, 0.7784, 0.7781, 0.7779, 0.7777, 0.7776, 0.7776,
        0.7774, 0.7768, 0.7763, 0.7761, 0.7760, 0.7759, 0.7757, 0.7757, 0.7757,
        0.7751, 0.7749, 0.7748, 0.7746, 0.7744, 0.7742, 0.7741, 0.7739, 0.7738,
        0.7735, 0.7735, 0.7735, 0.7734, 0.7733, 0.7732, 0.7729, 0.7728, 0.7724,
        0.7723, 0.7722, 0.7721, 0.7720, 0.7719, 0.7718, 0.7716, 0.7715, 0.7714,
        0.7713, 0.7711, 0.7709, 0.7708, 0.7705, 0.7704, 0.7697, 0.7697, 0.7697,
        0.7697, 0.7696, 0.7692, 0.7691, 0.7691, 0.7689, 0.7689, 0.7688, 0.7688,
        0.7686, 0.7682, 0.7677, 0.7675, 0.7674, 0.7673, 0.7673, 0.7671, 0.7669,
        0.7667, 0.7666, 0.7664, 0.7664, 0.7663, 0.7663, 0.7662, 0.7660, 0.7659,
        0.7656, 0.7655, 0.7654, 0.7654, 0.7650, 0.7650, 0.7649, 0.7649, 0.7647,
        0.7645, 0.7645, 0.7644, 0.7644, 0.7644, 0.7642, 0.7642, 0.7639, 0.7636,
        0.7636, 0.7636, 0.7636, 0.7633, 0.7632, 0.7631, 0.7629, 0.7627, 0.7625,
        0.7624, 0.7623, 0.7622, 0.7620, 0.7620, 0.7618, 0.7617, 0.7615, 0.7614,
        0.7613, 0.7613, 0.7609, 0.7608, 0.7607, 0.7607, 0.7607, 0.7606, 0.7595,
        0.7594, 0.7593, 0.7592, 0.7592, 0.7592, 0.7590, 0.7589, 0.7588, 0.7588,
        0.7588, 0.7585, 0.7584, 0.7584, 0.7584, 0.7583, 0.7582, 0.7582, 0.7581,
        0.7581, 0.7581, 0.7580, 0.7580, 0.7579, 0.7579, 0.7575, 0.7574, 0.7568,
        0.7567, 0.7567, 0.7561, 0.7560, 0.7560, 0.7559, 0.7558, 0.7554, 0.7554,
        0.7553, 0.7551, 0.7551, 0.7549, 0.7549, 0.7548, 0.7548, 0.7547, 0.7546,
        0.7545, 0.7544, 0.7543, 0.7543, 0.7542, 0.7539, 0.7536, 0.7534, 0.7532,
        0.7532, 0.7532, 0.7531, 0.7531, 0.7530, 0.7528, 0.7528, 0.7528, 0.7527,
        0.7526, 0.7526, 0.7524, 0.7523, 0.7521, 0.7521, 0.7519, 0.7519, 0.7518,
        0.7517, 0.7515, 0.7514, 0.7514, 0.7513, 0.7512, 0.7511, 0.7510, 0.7509,
        0.7509, 0.7508, 0.7508, 0.7501, 0.7501, 0.7500, 0.7498, 0.7493, 0.7491,
        0.7490, 0.7489, 0.7489, 0.7488, 0.7487, 0.7487, 0.7487, 0.7486, 0.7486,
        0.7486, 0.7485, 0.7485, 0.7483, 0.7483, 0.7482, 0.7479, 0.7478, 0.7477,
        0.7477, 0.7476, 0.7476, 0.7473, 0.7472, 0.7472, 0.7471, 0.7471, 0.7470,
        0.7470]

barrier = [0.5957, 0.5957, 0.5956, 0.5956, 0.5956, 0.5955, 0.5955, 0.5954, 0.5953,
        0.5949, 0.5946, 0.5945, 0.5945, 0.5941, 0.5939, 0.5933, 0.5932, 0.5929]

pedestrian = [0.7039, 0.7027, 0.7016, 0.6997, 0.6994, 0.6990, 0.6986, 0.6985, 0.6982,
        0.6979, 0.6978, 0.6969, 0.6968, 0.6962, 0.6956, 0.6953, 0.6944, 0.6942,]





import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


maxlen = max(max(len(car), len(truck)), max(len(pedestrian), len(barrier)))

car = car + [-1] *(maxlen - len(car))
truck = truck + [-1] *(maxlen - len(truck))
barrier = barrier + [-1] *(maxlen - len(barrier))
pedestrian = pedestrian + [-1] *(maxlen - len(pedestrian))



OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

# '#E3EAB2' '#F3F3CF'
# Define a custom colormap
custom_colors = ['#F3F3CF','#BFDDB6', '#56BBC5', '#377BB3', '#3C4376']  
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

# Define categories and Z values (without "Background")
# categories = ['Car', 'Truck', 'Construction_vehicle', 'Bus', 'Trailer', 'Barrier',
#               'Motorcycle', 'Bicycle', 'Pedestrian', 'Traffic_cone']  # Removed 'Background'

categories = ['Car', 'Truck', 'pedestrian', 'Barrier']  # Removed 'Background'


Z = np.stack([np.array(car).T, np.array(truck).T,  \
              np.array(pedestrian).T, np.array(barrier).T],axis=1) 

# Define y-axis grid (10 divisions from 0 to 1)
y_grid = np.linspace(0, 1, 201)

# Count points per grid cell
point_counts = np.zeros((len(y_grid) - 1, len(categories)))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        bin_index = np.digitize(Z[i, j], y_grid) - 1
        if 0 <= bin_index < len(y_grid) - 1:
            point_counts[bin_index, j] += 1

# Normalize point counts for color intensity
max_points = point_counts.max()
normalized_counts = point_counts / max_points

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8, 6))

# Draw background grid and adjust intensity
for i in range(len(y_grid) - 1):
    for j in range(len(categories)):
        intensity = normalized_counts[i, j]
        color = custom_cmap(intensity)
        y_min = y_grid[i]
        y_max = y_grid[i + 1]
        ax.add_patch(plt.Rectangle((j, y_min), 1, y_max - y_min, color=color, ec=None))

# Set correct extent for the last column
plt.xlim(0, len(categories))
plt.ylim(0, 1)

# Configure axis labels and ticks
plt.xticks(ticks=np.arange(len(categories)) + 0.5, labels=categories, rotation=90, fontsize=10)  # Align ticks with categories
# plt.yticks(ticks=y_grid, labels=np.round(y_grid, 2), fontsize=10)  # y-axis grid labels
# plt.yticks([])  # Remove y-axis ticks and labels
# plt.xlabel('Categories', fontsize=12)
# plt.ylabel('Semantic Score', fontsize=12)



# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm)
# cbar.set_label('Point Density', fontsize=12)
cbar.set_ticks([]) 


# Adjust layout and display the plot
plt.tight_layout()

# Save the figure
plt.savefig('./vis/semantic.png', dpi=300, bbox_inches='tight')

plt.show()
