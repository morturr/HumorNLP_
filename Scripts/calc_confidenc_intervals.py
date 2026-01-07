import pandas as pd
import numpy as np
from scipy.stats import t
import os


def calculate_ci_from_csv(file_path, n_seeds=4, confidence_level=0.95):
    """
    Reads a CSV file, calculates the 95% Confidence Interval (CI) for accuracy,
    and returns a summary table.
    """
    # 1. Read and Filter the Data
    try:
        # Read the CSV file
        df_raw = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Filter for only 'accuracy' and 'std' metrics
    df_filtered = df_raw[df_raw['metric'].isin(['accuracy', 'std'])].copy()

    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print("Error: DataFrame is empty after filtering for 'accuracy' and 'std'.")
        return None

    # Identify the columns containing the performance values (amazon, dadjokes, etc.)
    # We assume all columns after 'param_metric' are the test datasets.
    value_cols = df_filtered.columns[df_filtered.columns.get_loc('param_metric') + 1:]

    # 2. Reshape the DataFrame
    df_melt = df_filtered.melt(
        id_vars=['trained on', 'metric'],
        value_vars=value_cols,
        var_name='Tested On',
        value_name='value'
    )

    df_pivot = df_melt.pivot_table(
        index=['trained on', 'Tested On'],
        columns='metric',
        values='value'
    ).reset_index()

    # Ensure accuracy and std columns are numeric
    df_pivot['accuracy'] = pd.to_numeric(df_pivot['accuracy'], errors='coerce')
    df_pivot['std'] = pd.to_numeric(df_pivot['std'], errors='coerce')

    # Rename columns for clarity
    df_pivot.columns = ['Trained On', 'Tested On', 'Accuracy', 'Std']

    # --- 3. Calculate Confidence Interval Parameters ---

    # Degrees of Freedom (df)
    df = n_seeds - 1
    alpha = 1 - confidence_level

    # Critical t-Value: Using the t.ppf for the two-tailed critical value.
    t_critical = t.ppf(1 - alpha / 2, df)

    print(f"--- Statistical Parameters ---")
    print(f"Seeds (n): {n_seeds}")
    print(f"Degrees of Freedom (df): {df}")
    print(f"Confidence Level: {confidence_level * 100:.0f}%")
    print(f"Critical t-Value: {t_critical:.3f}")
    print("-" * 30)

    # Calculate Standard Error (SE)
    df_pivot['SE'] = df_pivot['Std'] / np.sqrt(n_seeds)

    # Calculate Margin of Error (ME)
    df_pivot['Margin of Error (ME)'] = t_critical * df_pivot['SE']

    # 4. Calculate 95% CI Bounds
    df_pivot['CI Lower Bound'] = df_pivot['Accuracy'] - df_pivot['Margin of Error (ME)']
    df_pivot['CI Upper Bound'] = df_pivot['Accuracy'] + df_pivot['Margin of Error (ME)']

    # Format the Margin of Error for clean reporting
    df_pivot['95% CI (ME)'] = '±' + df_pivot['Margin of Error (ME)'].round(4).astype(str)

    # Select and reorder final columns for presentation
    report_df = df_pivot[[
        'Trained On',
        'Tested On',
        'Accuracy',
        '95% CI (ME)',
        'CI Lower Bound',
        'CI Upper Bound'
    ]].round(4)

    return report_df


# --- EXECUTION ---
# Use the file name of your uploaded CSV file
file_name = 'llama LOO with std final results 2025-07-28.xlsx'
folder_path = '../Llama-2/Results/Tables/Finals/'
final_results = calculate_ci_from_csv(folder_path+file_name)

if final_results is not None:
    # Print the final table in Markdown format
    print("--- Confidence Interval Results (95% CI) ---")
    print(final_results.to_markdown(index=False))

    # Save the results to a new CSV file
    output_file_name = 'llama_ci_loo_results_script.csv'
    output_folder_path = folder_path + 'Confidence Intervals/'
    os.makedirs(output_folder_path, exist_ok=True)
    final_results.to_csv(output_folder_path + output_file_name, index=False)
    print(f"\nResults saved to '{output_file_name}'")