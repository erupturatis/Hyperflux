import os
import json
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import FuncFormatter

def parse_filename(filename):
    """
    Parses a filename like 'cifar10_resnet50_adam_highlr_2.json'
    to extract model configuration details.
    """
    # Remove the .json extension and split by underscore
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    # Expected format: {dataset}_{network}_{optimizer}_{lr_setting}_{version}
    # Example parts: ['cifar10', 'resnet50', 'adam', 'highlr', '2']
    if len(parts) < 4:
        # Fallback for unexpected filename formats
        return base_name

    dataset = parts[0]
    network = parts[1]
    optimizer = parts[2]
    lr_setting = parts[3] # CORRECTED: Was parts[4]

    # --- Improved label formatting ---
    # Format dataset
    if 'cifar100' in dataset:
        dataset_text = 'CIFAR-100'
    elif 'cifar10' in dataset:
        dataset_text = 'CIFAR-10'
    else:
        dataset_text = dataset.upper()

    # Format network
    if 'resnet' in network:
        network_text = network.replace('resnet', 'ResNet')
    elif 'vgg' in network:
        network_text = network.upper()
    else:
        network_text = network.capitalize()

    # Format optimizer
    optimizer_text = optimizer.upper()

    # Format learning rate setting
    lr_text = "High LR" if 'highlr' in lr_setting else "Low LR"
    
    # Create a clean label for the legend
    # e.g., "CIFAR-10, ResNet50, ADAM, High LR"
    label = f"{dataset_text}, {network_text}, {optimizer_text}, {lr_text}"
    
    return label

def plot_density_graphs_updated():
    """
    Finds all JSON files, reads the density data, and plots density vs. epochs
    on a single graph with a logarithmic Y-axis and saves as a PDF.
    """
    # Find all .json files in the directory
    json_files = glob.glob('*.json')
    
    if not json_files:
        print("No .json files found in the current directory.")
        return

    # --- START: Font size configuration from the first script ---
    AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_FONT_SIZE = 24, 18, 16
    # --- END: Font size configuration ---

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Process and plot data from each file
    for file_path in sorted(json_files):
        try:
            # Load the data array from the JSON file
            with open(file_path, 'r') as f:
                density_percentages = json.load(f)
            
            # X-axis: epochs (assuming index + 1 is the epoch number)
            epochs = range(1, len(density_percentages) + 1)
            
            # Y-axis: density data
            y_values = density_percentages
            
            # Get the label for the legend from the filename
            legend_label = parse_filename(os.path.basename(file_path))
            
            # Plot the data
            ax.plot(epochs, y_values, marker='o', linestyle='-', markersize=4, label=legend_label)

        except (json.JSONDecodeError, IndexError, FileNotFoundError) as e:
            print(f"Could not process file {file_path}: {e}")

    # --- MODIFICATIONS FOR LABEL SIZES ---
    
    # 1. Set Y-axis to logarithmic scale
    ax.set_yscale('log')

    # 2. Set labels and legend using the new font sizes
    ax.set_xlabel('Epoch', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Density (%)', fontsize=AXIS_LABEL_SIZE) # Added (%) to match first script style
    ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE) # Changed loc to 'best' for better placement

    # 3. Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    # Optional: Format Y-axis ticks to show a '%' sign, like in the first script
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%'))

    # Set axis limits for clarity
    ax.set_xlim(left=0)

    # --- MANUAL BORDER OVERRIDE (from first script for identical style) ---
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
        spine.set_visible(True)
    # --- END OF OVERRIDE ---
    
    plt.tight_layout()
    
    # Save the plot as a PDF file
    output_filename_pdf = 'density_vs_epochs.pdf'
    plt.savefig(output_filename_pdf, bbox_inches='tight', dpi=300)
    print(f"Plot saved as '{os.path.abspath(output_filename_pdf)}'")

    # Display the plot
    plt.show()

if __name__ == '__main__':
    plot_density_graphs_updated()