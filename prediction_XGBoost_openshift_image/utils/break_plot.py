import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 6), gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=0.05)  # Adjust space between plots

# Define your data and labels
xs_label = ["3", "6", "60", "120", "300", "3000\n(DeltaSherlock)\n(XGBoost)", "3\n(Praxi)\n(VW)", "3000\n(Praxi)\n(VW)"]
indices = np.arange(len(xs_label))
ys = np.random.rand(8) * 10000  # Example high range data

# Plot on both subplots
ax.bar(indices, ys, color='skyblue')
ax2.bar(indices, ys, color='skyblue')

# Define the range to skip
ax.set_ylim(8000, 10000)  # upper plot shows the high values
ax2.set_ylim(0, 2000)  # lower plot shows the low values

# Hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# Add diagonal lines to indicate the break in the plot
d = 0.015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# Set x-ticks and labels
ax2.set_xticks(indices)
ax2.set_xticklabels(xs_label, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('./test.pdf', bbox_inches='tight')
plt.close()