import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define a professional template for the plots
plt.style.use('seaborn-v0_8-paper')  # Clean base style for publications

# Customize additional parameters for a professional appearance
def set_plot_style():
    plt.rcParams.update({
        'font.size': 10,                  # General font size
        'font.family': 'serif',           # Use serif fonts
        'text.usetex': False,             # Use LaTeX if needed for labels
        'axes.titlesize': 12,             # Title font size
        'axes.labelsize': 10,             # Axis label font size
        'axes.linewidth': 0.8,            # Axis line width
        'xtick.labelsize': 8,             # X-axis tick label font size
        'ytick.labelsize': 8,             # Y-axis tick label font size
        'legend.fontsize': 8,             # Legend font size
        'legend.frameon': True,           # Add a frame to the legend
        'legend.framealpha': 0.9,         # Legend transparency
        'legend.loc': 'best',             # Automatically place legend
        'lines.linewidth': 1.5,           # Line width
        'lines.markersize': 6,            # Marker size
        'grid.color': 'gray',             # Grid color
        'grid.alpha': 0.6,                # Grid transparency
        'grid.linewidth': 0.5,            # Grid line width
        'figure.figsize': (6.4, 4.8),     # Default figure size
        'savefig.dpi': 300,               # High resolution for saving figures
        'savefig.format': 'pdf',          # Save figures as PDF by default
    })

# Apply the style
set_plot_style()

# Example plot
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

fig, ax = plt.subplots()

# Plot data
ax.plot(x, y, label='Example Line', color='blue', marker='o')

# Titles and labels
ax.set_title('Professional Research Plot Example')
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')

# Add a grid
ax.grid(True)

# Customize ticks
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.1f}'))

# Add legend
ax.legend()

# Tight layout for better spacing
plt.tight_layout()

# Save or show the plot
plt.savefig('research_plot.pdf')  # PDF format
plt.savefig('research_plot.svg')  # SVG format
plt.savefig('research_plot.eps')  # EPS format
plt.show()

# Save the figure in different vector formats
