import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import os
import argparse
from matplotlib.ticker import FuncFormatter

# --- 1. Model Definitions for Curve Fitting ---

def power_law_model(x, c, alpha):
    """
    A power-law model: S(γ) = c * γ^(-α)
    This describes a linear relationship in a log-log plot.
    S: Sparsity (Remaining %)
    γ: Saliency
    """
    return c * (x ** -alpha)

def linear_model_log(log_x, b, m):
    """
    A linear model for log-transformed data: log(y) = b + m * log(x)
    b: y-intercept (which is log(c) in the power law)
    m: slope (which is -alpha in the power law)
    """
    return b + m * log_x


# --- 2. Data Loading Function ---
def read_pruning_data(csv_filepath, saliency_col, remaining_col):
    """
    Reads a CSV file and extracts saliency and remaining weight percentage columns.
    """
    saliency_scores = []
    remaining_percentages = []

    if not os.path.isfile(csv_filepath):
        print(f"Error: The file '{csv_filepath}' was not found.")
        return None, None

    with open(csv_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if saliency_col not in reader.fieldnames or remaining_col not in reader.fieldnames:
            print(f"Error: One or both specified columns were not found in the CSV.")
            print(f"  - Expected Saliency Column: '{saliency_col}'")
            print(f"  - Expected Remaining Weights Column: '{remaining_col}'")
            print(f"  - Available columns: {', '.join(reader.fieldnames)}")
            return None, None
        for row in reader:
            try:
                saliency_scores.append(float(row[saliency_col]))
                remaining_percentages.append(float(row[remaining_col]))
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process row in CSV: {row}. Error: {e}")
                continue
    return saliency_scores, remaining_percentages

# --- 3. Main Execution Block ---
def main():
    """
    Main function to parse arguments, load data, perform curve fitting on a
    specific region, and generate the plot.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Plot sparsity vs. saliency, highlighting a specific region and fitting a power-law curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--file', '-f', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--saliency_col', '-sa', type=str, default='Saliency', help="Name of the column for the saliency score.")
    parser.add_argument('--remaining_col', '-sp', type=str, default='Remaining', help="Name of the column for the remaining weights percentage.")
    args = parser.parse_args()

    # --- USER CONFIGURATIONS ---
    # Set the percentage boundaries (r1, r2) for the three regions.
    # Region 1: (r1, 100%]
    # Region 2: (r2, r1]   <-- This is the region of interest for the fit
    # Region 3: [0, r2]
    R1_PERCENT = 50.0
    R2_PERCENT = 0.5

    # --- Plot Styling Configuration ---
    AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE, MARKER_SIZE = 16, 14, 12, 18, 60

    # --- Data Loading ---
    saliency_scores, remaining_percentages = read_pruning_data(args.file, args.saliency_col, args.remaining_col)
    if saliency_scores is None:
        return

    saliency_scores = np.array(saliency_scores, dtype=float)
    remaining_percentages = np.array(remaining_percentages, dtype=float)

    # --- Data Segmentation into Three Regions ---
    # Create boolean masks to identify data points in each region based on remaining %
    region1_mask = (remaining_percentages > R1_PERCENT) & (remaining_percentages <= 100)
    region2_mask = (remaining_percentages > R2_PERCENT) & (remaining_percentages <= R1_PERCENT)
    region3_mask = (remaining_percentages <= R2_PERCENT)

    # --- Curve Fitting (Power-Law on Region 2) ---
    # Filter data for Region 2: saliency and sparsity must be positive for log scale fitting
    fit_mask = (saliency_scores > 0) & (remaining_percentages > 0) & region2_mask

    threshold_curve, sparsity_curve = None, None
    if np.sum(fit_mask) < 2:
        print(f"Warning: Not enough valid data points in Region 2 (between {R2_PERCENT}% and {R1_PERCENT}%) for curve fitting. Need at least 2 points.")
    else:
        saliency_fit = saliency_scores[fit_mask]
        sparsity_fit = remaining_percentages[fit_mask]
        
        try:
            # --- MODIFIED SECTION: Fit in log space ---
            log_saliency = np.log(saliency_fit)
            log_sparsity = np.log(sparsity_fit)

            # Fit the linear model to the log-transformed data
            p0 = [np.log(sparsity_fit[0]), -1.0] # Initial guess for [b, m]
            popt_log, _ = curve_fit(linear_model_log, log_saliency, log_sparsity, p0=p0)
            b_fit, m_fit = popt_log

            # Convert linear fit parameters back to power-law parameters
            c_fit = np.exp(b_fit)
            alpha_fit = -m_fit
            # --- END OF MODIFIED SECTION ---

            print("--- Power-Law Fit Results for Region 2 (Fitted in Log Space) ---")
            print(f"Model: S(γ) = c * γ^(-α)")
            print(f"Fitted parameters: c = {c_fit:.4f}, α = {alpha_fit:.4f}")

            # Generate the curve using the fitted model, spanning the saliency range of Region 2
            min_saliency = saliency_fit.min()
            max_saliency = saliency_fit.max()
            threshold_curve = np.logspace(np.log10(min_saliency), np.log10(max_saliency), 200)
            sparsity_curve = power_law_model(threshold_curve, c_fit, alpha_fit)

        except Exception as e:
            print(f"Curve fitting failed: {e}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Plot the three regions with different styles
    # Region 1 (semi-transparent)
    plt.scatter(saliency_scores[region1_mask], remaining_percentages[region1_mask],
                s=MARKER_SIZE, label=f"Region 1 (> {R1_PERCENT}%)", color='gray', alpha=0.4, zorder=3)

    # Region 2 (main focus)
    plt.scatter(saliency_scores[region2_mask], remaining_percentages[region2_mask],
                s=MARKER_SIZE, label=f"Region 2 ({R2_PERCENT}% - {R1_PERCENT}%)", color='blue', alpha=0.9, zorder=5)

    # Region 3 (semi-transparent)
    plt.scatter(saliency_scores[region3_mask], remaining_percentages[region3_mask],
                s=MARKER_SIZE, label=f"Region 3 (< {R2_PERCENT}%)", color='gray', alpha=0.4, zorder=3)

    # Plot the fitted curve if it was successfully generated
    if threshold_curve is not None:
        plt.plot(threshold_curve, sparsity_curve, linestyle='-', linewidth=2.5,
                 label=f"Power-Law Fit (Region 2)\n$S = {c_fit:.2f} \\cdot \\gamma^{{-{alpha_fit:.2f}}}$",
                 color='red', alpha=0.8, zorder=10)

    # --- Axes and Labels ---
    plt.xscale('log')
    plt.yscale('log')
    
    xlabel = args.saliency_col.replace('_', ' ').title()
    ylabel = "Remaining Weights (%)"
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    plt.title(f"Sparsity vs. {xlabel}", fontsize=TITLE_FONT_SIZE, pad=20)
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%'))
    plt.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    plot_filename = "sparsity_vs_saliency_regional_fit.pdf"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot successfully saved to {os.path.abspath(plot_filename)}")
    plt.show()

if __name__ == "__main__":
    main()