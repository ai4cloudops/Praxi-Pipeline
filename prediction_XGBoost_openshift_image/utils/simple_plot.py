import matplotlib.pyplot as plt

# Paste your data list here
data = [73.202872, 28.350209, 29.701616, 28.789624, 31.211933, 31.871822, 39.362723, 31.974545, 33.669389, 17.346602, 36.856092, 40.076779, 40.088666, 90.510416, 91.792845, 22.315254, 22.254104, 44.000223, 70.510059, 22.802672, 44.82052, 23.497387, 46.165653, 23.575701, 47.043408, 47.807015, 24.418639, 541.842788, 50.156214, 50.736789, 50.642727, 50.75856, 26.495919, 84.864985, 83.376989, 26.464443, 55.377038, 57.559413, 60.322053, 93.007852, 57.466134, 57.282262, 29.39939, 87.073757, 58.594705, 57.456598, 29.91736, 58.754895, 59.760215, 59.093035, 61.424267, 29.916725, 60.920218, 43.578147, 63.386553, 62.911737, 99.219213, 63.530364, 63.82445, 63.819183, 65.272151, 64.33446, 64.974107, 98.362624, 65.103753, 34.569741, 65.864989, 66.44651, 128.268032, 64.169447, 64.147027, 87.4692, 34.875772, 67.227295, 112.808594, 67.827126, 70.640686, 35.158679, 36.564811, 68.495763, 69.352932, 34.928663, 72.169425, 34.903185, 70.259288, 69.93001, 190.769735, 70.273503, 71.031678, 35.911438, 35.893764, 71.573536, 72.284772, 71.652171, 36.548442, 107.009302, 71.967316, 36.476078, 106.581399, 69.889697, 70.208841, 35.806549, 70.417283, 36.202654, 126.59347, 69.924674, 35.877587, 71.369456, 72.0081, 71.102271, 205.618984, 108.385519, 110.0021, 111.298061, 36.141017, 37.549446, 37.261468, 73.569084, 37.623378, 38.00455, 74.611679, 75.482532, 74.377725, 76.099858, 74.84326, 75.398062, 74.90488, 75.34521, 278.805906, 37.626238, 36.649223, 37.187985, 37.237206, 74.876472, 38.210068, 75.339293, 75.872817, 38.161517, 74.776258, 75.027992, 76.248884, 38.439168, 39.345977, 76.888348, 37.977319, 38.288083, 63.711843, 75.461236, 76.415971, 76.608811, 38.245052, 76.568717, 39.521801, 38.98797, 93.247591, 76.976929, 159.602679, 76.243201, 115.469697, 78.973834, 173.90518, 40.605625, 78.386021, 125.198808, 78.636335, 221.168661, 78.667635, 39.878624, 78.718324, 78.988829, 78.274549, 110.372428, 161.18218, 40.081144, 77.940861, 78.124584, 117.71202, 78.031484, 78.900211, 75.833933, 79.243125, 40.266875, 39.732143, 79.883278, 79.916352, 79.756462, 104.604326, 40.131956, 47.658164, 40.298139, 40.39604, 79.676014, 39.884777, 76.945253, 39.6631, 40.635219, 88.787061, 39.038272, 39.015213, 237.66375, 80.093619, 80.251103, 117.765876, 78.164418, 78.585706, 77.570943, 38.847509, 80.541437, 41.369952, 42.104175, 80.877567, 80.874378, 40.815249, 81.449937, 156.686269, 41.44103, 81.154621, 121.692868, 123.203419, 80.520642, 41.712489, 79.151789, 51.11959, 40.278764, 81.153044, 71.351992, 81.22313, 41.64632, 134.449975, 82.111636, 78.921098, 81.650361, 134.509238, 41.473838, 41.654231, 245.264345, 125.496069, 82.06254, 126.240293, 205.01975, 41.787211, 42.078709, 82.505688, 82.536019, 136.291718, 128.177248, 82.456717, 118.014517, 41.97205, 81.918243, 79.849055, 58.052114, 50.871289, 90.096849, 81.630581, 81.590321, 82.337938, 81.582028, 83.20304, 81.884045, 1151.92123, 78.744017, 39.784155, 40.342288, 88.882228, 78.748374, 168.095621, 79.100289, 39.750158, 39.700547, 39.8041, 79.462934, 78.264413, 78.210754, 78.343009, 78.619135, 40.026474, 120.281815, 78.709969, 81.318291, 79.197823, 40.59551, 41.275768, 79.202378, 78.992398, 80.941871, 79.697968, 79.06288, 40.744137, 277.265428, 40.634996, 79.60691, 80.455898, 40.711, 79.72607, 41.121943, 40.838363, 82.619751, 80.571429, 121.950577, 81.186747, 41.076113, 81.690301, 41.238995, 105.496777, 163.736171, 81.477006, 41.209677, 81.704252, 81.70993, 173.173326, 81.163642, 212.01594, 41.562488, 81.483624, 41.282779, 40.775481, 176.19514, 41.59628, 82.059056, 81.933943, 164.5115, 80.558054, 81.578149, 41.192786, 64.150546, 80.951296, 81.79687, 123.992035, 82.437533, 81.541792, 80.969818, 41.765899, 41.507754, 82.10332, 81.404437, 81.46497, 41.570133, 279.532899, 82.020679, 125.01828, 88.750336, 128.202564, 41.915846, 83.273854, 41.548845, 81.841482, 174.402385, 42.008359, 82.640553, 41.830971, 91.134694, 42.331504, 138.408411, 41.912894, 42.840994, 42.308663, 42.19436, 212.291937, 587.116539, 83.009546, 42.144207, 42.139713, 82.372584, 83.310212, 42.245079, 61.19171, 125.464609, 42.171014, 83.202605, 42.425056, 54.217754, 91.441342, 124.505941, 42.199803, 368.956039, 83.438023, 82.774548, 42.088331, 83.124631, 42.337296, 83.586297, 42.25501, 42.621179, 126.122747, 152.390571, 41.98789, 125.332087, 42.642256, 84.876174, 83.461633, 83.22502, 42.477787, 83.138064, 84.544779, 42.104469, 42.904992, 84.569436, 83.266722, 85.198258, 42.889265, 42.439315, 42.556082, 42.560482, 42.903113, 44.086454, 83.674656, 83.494118, 84.785665, 128.763324, 84.592957, 85.063727, 131.294653, 83.912085, 84.256109, 127.704195, 43.203663, 84.735541, 84.750851, 42.937119, 190.35795, 84.742795, 173.55371, 83.795844, 85.680354, 43.193604, 43.256371, 85.027613, 85.939046, 85.285227, 43.084848, 85.012761, 42.961195, 84.096771, 42.92131, 86.613885, 84.433984, 42.904993, 85.014023, 84.9167, 151.585424, 85.441442, 84.714662, 130.744679, 89.728258, 84.792618, 188.899997, 84.842887, 342.736331, 85.776125, 85.015609, 43.541757, 44.33475, 43.702029, 84.594744, 84.89061, 85.844394, 84.651784, 130.848579, 132.567787, 84.460709, 202.784308, 84.89592, 43.536783, 85.9664, 238.32701, 84.671994, 85.329998, 194.939589, 108.798724, 43.685766, 85.481111, 43.629464, 43.525921, 85.575015, 85.917768, 43.668885, 130.912159, 85.112351, 84.890369, 43.779987, 43.846465, 43.771501, 43.636584, 43.435458, 129.737622, 43.909802, 86.107542, 92.744722, 43.744556, 86.872502, 85.863085, 85.324636, 85.555887, 86.006912, 85.528488, 85.161167, 43.937287, 43.204586, 44.033926, 43.117431, 43.691673, 86.747702, 85.632406, 43.365087, 88.481642, 86.818161, 185.488186, 43.529444, 85.925891, 86.754607, 174.190479, 88.804255, 43.671963, 86.784285, 141.506599, 43.56104, 43.537221, 43.898879, 44.076895, 44.363022, 86.058276, 192.424445, 86.73347, 87.529264, 44.66906, 86.487047, 131.958891, 128.623091, 43.642839, 43.695316, 85.809652, 44.026763, 43.820985, 44.119171, 44.109298, 43.693419, 44.213228, 44.508071, 175.447843, 44.306356, 87.093333, 86.950425, 43.704417, 44.182654, 88.641505, 44.490265, 55.685163, 130.716666, 44.410093, 87.313536, 44.188447, 87.581246, 43.895513, 87.040935, 87.057601, 87.215153, 176.284256, 133.586032, 88.00224, 122.752327, 87.665913, 44.327936, 87.457072, 44.067379, 87.199846, 87.358948, 44.231115, 86.879024, 43.960583, 87.76985, 87.069513, 45.01604, 181.484108, 80.32234, 44.448081, 87.179178, 130.929417, 44.094202, 88.235388, 88.664575, 44.273545, 44.318491, 88.026095, 181.968111, 87.229228, 44.791713, 87.888206, 87.915911, 87.446534, 87.02296, 87.213488, 45.000002, 87.76394, 123.100158, 190.735672, 45.174541, 88.105889, 89.392403, 87.854432, 269.183527, 45.259838, 90.617802, 44.591169, 44.874363, 136.257785, 88.780346, 44.677065, 89.606546, 160.842602, 87.79169, 88.558255, 44.869872, 87.672175, 131.813707, 152.945424, 132.634656, 90.640794, 89.72779, 182.406039, 94.232033, 89.365402, 115.739427, 87.437058, 45.38773, 134.552441, 144.141032, 87.449618, 44.774584, 88.307232, 134.894332, 44.656336, 44.629746, 46.861911, 44.662993, 136.755031, 88.735483, 89.546159, 81.196899, 45.365951, 88.930861, 44.900362, 45.366076, 44.721456, 45.44495, 87.733584, 44.981352, 44.920225, 88.591625, 87.930569, 185.836986, 132.331857, 136.859729, 44.866184, 44.782411, 45.553179, 88.276784, 101.164983, 89.215412, 88.869561, 88.127408, 88.681264, 45.287478, 89.778818, 45.132291, 44.442452, 88.254374, 88.994691, 89.00766, 45.144964, 45.1123, 88.544471, 91.144855, 45.338125, 90.03676, 88.847197, 88.138111, 89.290389, 88.974928, 44.660814, 45.085963, 88.69504, 45.421256, 88.666393, 249.091366, 137.424248, 45.439894, 89.060638, 45.305258, 89.601196, 45.382434, 89.484499, 181.875746, 44.992028, 141.852835, 45.264507, 154.630402, 45.97454, 45.027519, 134.283046, 45.396724, 45.876767, 183.099318, 89.051818, 45.422546, 90.33032, 89.010334, 88.70303, 138.983213, 89.15588, 170.886233, 45.354065, 142.976632, 45.34959, 45.638158, 45.287402, 183.740004, 89.319179, 431.811294, 88.946337, 45.172738, 89.961357, 90.069666, 90.515261, 139.68714, 47.584368, 163.246116, 90.424218, 90.065335, 90.05769, 92.706148, 91.058627, 45.968342, 89.386019, 89.763958, 183.454018, 90.184645, 90.125093, 54.114663, 91.876791, 91.780026, 46.008151, 90.170598, 89.902492, 89.254275, 90.551982, 45.328256, 45.767812, 90.161621, 132.241983, 97.012769, 45.468266, 46.034764, 45.795512, 46.080136, 89.596126, 89.203832, 153.759479, 90.149317, 89.725551, 89.910219, 56.222282, 498.81761, 90.899345, 90.342901, 90.016446, 45.956568, 45.293387, 140.952878, 45.517352, 89.720734, 46.685346, 90.259525, 45.585297, 90.474487, 90.381735, 89.852966, 89.712318, 90.540928, 45.734001, 90.220505, 135.29873, 137.104963, 90.739351, 181.232479, 46.07008, 46.170885, 45.799986, 91.919002, 46.072659, 46.377395, 137.430905, 90.624463, 160.436683, 45.432009, 90.779871, 90.398211, 90.521568, 91.424729, 45.490036, 92.549103, 46.026954, 90.782526, 45.881019, 90.420507, 45.523405, 53.801822, 45.566256, 90.320873, 91.020633, 91.427001, 90.273122, 46.508645, 136.597519, 45.994165, 183.692159, 46.149662, 90.001177, 46.87144, 93.639021, 137.012314, 90.157064, 143.896898, 46.397951, 107.285812, 46.428972, 46.36074, 90.409965, 90.50802, 46.182988, 46.316794, 91.641999, 91.404352, 90.838376, 91.097498, 91.103487, 135.151729, 67.411923, 46.683316, 46.153343, 106.417745, 91.512284, 46.325412, 89.056359, 65.703293, 136.367581, 142.648638, 91.770179, 46.331297, 90.806277, 100.350241, 46.165104, 142.581998, 92.393189, 63.08559, 46.363388, 91.301955, 91.703882, 91.514173, 46.727329, 46.37616, 91.033698, 91.578264, 93.926634, 46.309737, 90.653545, 66.709306, 91.134169, 90.463453, 91.645986, 45.965402, 55.995758, 91.111844, 97.558358, 46.489775, 91.917664, 92.534123, 91.547332, 47.008674, 141.589938, 91.708752, 90.939435, 46.87891, 93.561398, 90.475467, 91.937282, 91.602976, 62.051069, 138.599503, 46.35388, 46.413022, 46.462473, 46.31654, 50.612441, 91.930491, 101.288804, 91.651246, 46.5868, 91.884884, 91.333977, 46.825012, 46.425452, 92.387047, 46.57614, 92.105975, 91.496561, 91.837793, 46.548835, 137.037985, 91.663907, 46.673858, 91.741251, 100.642208, 91.527273, 92.556478, 91.910783, 103.136026, 92.047186, 54.48735, 46.759009, 46.843119, 92.414119, 92.803527, 47.166462, 92.683332, 138.218584, 166.480351, 47.010647, 91.49322, 46.831416, 91.512756, 93.098952, 92.258608, 101.43855, 46.704386, 92.05546, 92.202424, 143.373125, 92.112639, 92.164834, 92.390213, 92.552303, 92.510749, 47.329503, 46.781792, 46.934792, 92.806208, 91.943196, 91.009295, 153.967683, 115.348886, 46.620288, 46.693995, 47.162231, 91.741586, 91.806403, 47.46775, 92.823504, 46.753645, 47.078314, 46.429908, 47.138566, 479.935045, 46.542025, 92.630706, 92.862793, 46.953014, 93.194132, 93.572351, 46.826384, 47.153211, 198.286054, 92.234978, 138.186877, 46.546918, 93.303863, 92.046066, 56.278066, 91.358091, 46.791692, 91.740972, 93.009094, 92.400823, 94.774073, 138.777571, 58.43368, 47.146005, 156.563466, 47.280239, 46.92792, 47.674301, 47.529562, 92.500895, 92.417689, 92.346369, 91.886979, 47.447793, 47.11185, 145.673794, 93.230628]

# Create a figure and an axis
fig, ax = plt.subplots()

# Plot data
ax.plot(data, 'o')  # 'o' creates a scatter plot with circle markers

# Set the title and labels
ax.set_title('Data Points')
ax.set_xlabel('Index')
ax.set_ylabel('Value')

# Show the plot
# plt.show()
plt.savefig("./test.pdf", bbox_inches='tight')