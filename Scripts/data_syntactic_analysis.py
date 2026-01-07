import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mc
import colorsys

# Assuming 'FlanT5.data_loader' is accessible in your environment
from FlanT5.data_loader import load_dataset

# --- Global Plotting Constants ---

# Constants for Question Mark Count Analysis
MAX_Q_COUNT = 5
Q_BINS = np.arange(MAX_Q_COUNT + 2) - 0.5
Q_BIN_LABELS = [str(i) for i in range(MAX_Q_COUNT)] + [f'{MAX_Q_COUNT}+']
# PLOT_Q_COUNT = False # Changed to True to demonstrate the new plot
PLOT_Q_COUNT = True

# Constants for Text Length Analysis
MAX_LENGTH = 150  # Max character length to display
LENGTH_BIN_WIDTH = 10
LENGTH_BINS = np.arange(0, MAX_LENGTH + LENGTH_BIN_WIDTH, LENGTH_BIN_WIDTH)
# PLOT_TEXT_LENGTH = True # Assuming this remains True
PLOT_TEXT_LENGTH = True

# Constants for Word Count Analysis
MAX_WORD_COUNT = 35  # Max word count to display
WORDS_BIN_WIDTH = 5
WORD_COUNT_BINS = np.arange(0, MAX_WORD_COUNT + WORDS_BIN_WIDTH, WORDS_BIN_WIDTH)
# PLOT_WORD_COUNT = False # Changed to True to demonstrate the new plot
PLOT_WORD_COUNT = True

# Shared plotting style parameters
PLOT_STYLE_PARAMS = {
    'rwidth': 0.9,
    # 'edgecolor': 'black',
    'align': 'mid',
}

def darken_color(color, amount=0.5):
    """
    Darkens the given color by multiplying the lightness (L) by the given amount.
    Amount < 1.0 makes it darker, Amount > 1.0 makes it lighter.
    """
    try:
        c = mc.cnames[color] # Convert color name to hex/RGB
    except:
        c = color
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    # Adjust lightness (L) and convert back to RGB
    return colorsys.hls_to_rgb(h, max(0, min(1, amount * l)), s)

# --- NEW 4x3 Plotting Functions for All Features ---

def plot_4x3_grid(all_data_list, all_names, feature_name, x_label, bins, tick_labels, y_lims, title_suffix):
    """
    Generates a single figure with 4 rows (Datasets) and 3 columns (Total/Pos/Neg)
    for the specified feature analysis.
    """
    # Create a single figure with 4 rows and 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), constrained_layout=True, sharey=True)
    fig.suptitle(f'Comparative Distribution of {feature_name} Across All Datasets{title_suffix}', fontsize=18,
                 fontweight='bold')

    titles = ['Total Dataset', 'Positive Class (Humorous)', 'Negative Class (Non-Humorous)']

    # Configure plot parameters based on feature
    plot_kwargs = {'bins': bins, **PLOT_STYLE_PARAMS}

    # Check if x-axis labels need rotation
    rotation = 45 if len(tick_labels) > 10 else 0

    # Iterate through each dataset (row)
    for row_idx, dataset_name in enumerate(all_names):
        colors = [sns.color_palette("Set2")[row_idx]] * 3

        data_total, data_positive, data_negative = all_data_list[row_idx]

        # Iterate through the three classes (column)
        for col_idx, (data, title, color) in enumerate(zip(
                [data_total, data_positive, data_negative], titles, colors
        )):
            ax = axes[row_idx, col_idx]

            # Plot the histogram with different hatching for each class
            edgecolor = darken_color(color)
            if col_idx == 0:
                ax.hist(data, color=color, **plot_kwargs, edgecolor=edgecolor)
            elif col_idx == 1:
                ax.hist(data, color=color, **plot_kwargs, hatch='x', edgecolor=edgecolor)
            else:
                ax.hist(data, color=color, **plot_kwargs, hatch='.', edgecolor=edgecolor)

            # Apply customizations
            if row_idx == 0:
                # Set column titles only for the top row
                ax.set_title(title, fontsize=14)

            if col_idx == 0:
                # Set dataset name (row label) only for the first column
                ax.set_ylabel(f'Frequency ({dataset_name})', fontsize=12, fontweight='bold')

            if row_idx == 3:
                # Set x-axis label only for the bottom row
                ax.set_xlabel(x_label, fontsize=12)

            # Set X-ticks and limits
            if bins is Q_BINS:
                ax.set_xticks(range(MAX_Q_COUNT + 1))
            elif bins is LENGTH_BINS or bins is WORD_COUNT_BINS:
                ax.set_xticks(bins)

            ax.set_xticklabels(tick_labels, rotation=rotation)
            ax.set_ylim(y_lims)
            ax.grid(axis='y', alpha=0.5, linestyle='--')

    plt.show()


# --- Analysis Function (Individual plotting removed) ---

def analyze_dataset(df, dataset_name, text_col='text', label_col='label', positive_label=1, negative_label=0):
    """
    Performs the syntactic analysis and returns capped counts for combined analysis.
    (Individual plotting logic for Q-Count, Length, and Word-Count has been removed
    to be handled by the single 4x3 plot function.)
    """
    # 1A. Count the number of '?' in each sample
    df['question_mark_count'] = df[text_col].apply(lambda x: str(x).count('?'))

    # 1B. Calculate Text Length
    df['text_length'] = df[text_col].apply(lambda x: len(str(x)))

    # 1C. Calculate Word Count
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))

    # --- Prepare and Cap Question Mark data ---
    data_total_q = df['question_mark_count']
    data_positive_q = df[df[label_col] == positive_label]['question_mark_count']
    data_negative_q = df[df[label_col] == negative_label]['question_mark_count']
    data_total_capped_q = data_total_q.clip(upper=MAX_Q_COUNT)
    data_positive_capped_q = data_positive_q.clip(upper=MAX_Q_COUNT)
    data_negative_capped_q = data_negative_q.clip(upper=MAX_Q_COUNT)

    # --- Prepare and Cap Length data ---
    data_total_len = df['text_length']
    data_positive_len = df[df[label_col] == positive_label]['text_length']
    data_negative_len = df[df[label_col] == negative_label]['text_length']
    data_total_capped_len = data_total_len.clip(upper=MAX_LENGTH)
    data_positive_capped_len = data_positive_len.clip(upper=MAX_LENGTH)
    data_negative_capped_len = data_negative_len.clip(upper=MAX_LENGTH)

    # --- Prepare and Cap Word Count data ---
    data_total_words = df['word_count']
    data_positive_words = df[df[label_col] == positive_label]['word_count']
    data_negative_words = df[df[label_col] == negative_label]['word_count']
    data_total_capped_words = data_total_words.clip(upper=MAX_WORD_COUNT)
    data_positive_capped_words = data_positive_words.clip(upper=MAX_WORD_COUNT)
    data_negative_capped_words = data_negative_words.clip(upper=MAX_WORD_COUNT)

    # Return all capped data arrays for use in the combined plot
    return (data_total_capped_q.values, data_positive_capped_q.values, data_negative_capped_q.values,
            data_total_capped_len.values, data_positive_capped_len.values, data_negative_capped_len.values,
            data_total_capped_words.values, data_positive_capped_words.values, data_negative_capped_words.values)


# --- Main Execution Block (Modified for 4x3 plotting) ---

def main():
    datasets_info = {}
    dataset_codenames = ['amazon', 'dadjokes', 'headlines', 'one_liners']
    dataset_names = ['Amazon', 'Dad Jokes', 'Headlines', 'One Liners']

    # Lists to hold (total, positive, negative) data for all 4 datasets
    all_dataset_q_data = []
    all_dataset_len_data = []
    all_dataset_words_data = []

    for i, dataset in enumerate(dataset_codenames):
        # NOTE: load_dataset function call remains the same
        data = load_dataset(dataset_name=dataset, test_percent=1125.0,
                            add_instruction=False, instruction_version=0, with_val=False,
                            train_size=5000)

        if dataset == 'amazon':
            print('*** Removing product names from Amazon dataset... ***')
            remove_product = lambda example: {
                'text': example['text'][example['text'].index(' [SEP] ') + len(' [SEP] '):]}
            for split in ['train', 'test']:
                data[split] = data[split].map(
                    remove_product,
                    batched=False
                )

        datasets_info[dataset_names[i]] = data['train'].to_pandas()

    # Run analysis and collect all counts
    for name, df in datasets_info.items():
        (total_q, positive_q, negative_q,
         total_len, positive_len, negative_len,
         total_words, positive_words, negative_words) = analyze_dataset(
            df=df,
            dataset_name=name,
            text_col='text',
            label_col='label',
            positive_label=1,
            negative_label=0
        )

        # Collect data for each feature
        all_dataset_q_data.append((total_q, positive_q, negative_q))
        all_dataset_len_data.append((total_len, positive_len, negative_len))
        all_dataset_words_data.append((total_words, positive_words, negative_words))

    # --- Generate the NEW 4x3 Combined Plot for Question Mark Count ---
    if PLOT_Q_COUNT:
        plot_4x3_grid(
            all_dataset_q_data,
            dataset_names,
            feature_name='Question Mark (?) Count',
            x_label='Number of Question Marks (?) per Sample',
            bins=Q_BINS,
            tick_labels=Q_BIN_LABELS,
            y_lims=(0, 5300),  # Leave y_lims empty for auto-scaling
            title_suffix=' (Capped at 5+)'
        )

    # --- Generate the NEW 4x3 Combined Plot for Text Length ---
    if PLOT_TEXT_LENGTH:
        plot_4x3_grid(
            all_dataset_len_data,
            dataset_names,
            feature_name='Text Length (Characters)',
            x_label='Text Length (Characters)',
            bins=LENGTH_BINS,
            tick_labels=[f'{int(b)}' for b in LENGTH_BINS],
            y_lims=(0, 1500),  # Using the explicit y_lim from the original code
            title_suffix=f' (Capped at {MAX_LENGTH}+)'
        )

    # --- Generate the NEW 4x3 Combined Plot for Word Count ---
    if PLOT_WORD_COUNT:
        plot_4x3_grid(
            all_dataset_words_data,
            dataset_names,
            feature_name='Word Count',
            x_label='Word Count',
            bins=WORD_COUNT_BINS,
            tick_labels=[f'{int(b)}' for b in WORD_COUNT_BINS],
            y_lims=(0, 2500),  # Leave y_lims empty for auto-scaling
            title_suffix=f' (Capped at {MAX_WORD_COUNT}+)'
        )


if __name__ == '__main__':
    main()
    pass
    print("We're done with all 4x3 comparative plots!")