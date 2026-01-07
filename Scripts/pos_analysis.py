import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('punkt')
from collections import defaultdict
import seaborn as sns

from FlanT5.data_loader import load_dataset


# NOTE: You must ensure the necessary NLTK models are downloaded before running:
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# --- 2. POS Tagging and Mapping Logic ---

# Define the major POS categories we want to analyze
POS_CATEGORIES = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Pronoun', 'Determiner', 'Other']
LEXICAL_WORDS = ['Noun', 'Verb', 'Adjective', 'Adverb']


def map_pos_tag(tag):
    """Maps detailed Penn Treebank tags to broad categories."""
    if tag.startswith('NN'):
        return 'Noun'
    elif tag.startswith('VB'):
        return 'Verb'
    elif tag.startswith('JJ'):
        return 'Adjective'
    elif tag.startswith('RB'):
        return 'Adverb'
    elif tag.startswith('PRP') or tag.startswith('WP'):  # Pronoun tags
        return 'Pronoun'
    elif tag.startswith('DT') or tag.startswith('WDT'):  # Determiners
        return 'Determiner'
    else:
        return 'Other'  # Includes punctuation, conjunctions, interjections (UH), etc.


def calculate_pos_metrics(text):
    """
    Tokenizes, tags, and calculates the percentage of each major POS category
    and the Lexical Density for a single text sample.
    Returns a dictionary of metrics.
    """
    if not text:
        return {pos: 0.0 for pos in POS_CATEGORIES + ['Lexical Density']}

    # Tokenize and Tag
    tokens = nltk.word_tokenize(str(text).lower())
    tagged_words = nltk.pos_tag(tokens)

    # Count tags and categorize
    pos_counts = defaultdict(int)
    total_words = 0
    total_lexical_words = 0

    for word, tag in tagged_words:
        # Ignore punctuation/non-alphanumeric tokens for a cleaner count
        if word.isalnum():
            total_words += 1
            mapped_tag = map_pos_tag(tag)
            pos_counts[mapped_tag] += 1

            if mapped_tag in LEXICAL_WORDS:
                total_lexical_words += 1

    if total_words == 0:
        return {pos: 0.0 for pos in POS_CATEGORIES + ['Lexical Density']}

    # Calculate percentages
    metrics = {tag: (count / total_words) * 100 for tag, count in pos_counts.items()}

    # Calculate Lexical Density (Analysis #2)
    metrics['Lexical Density'] = (total_lexical_words / total_words) * 100

    # Ensure all defined categories exist, even if count is 0
    for pos in POS_CATEGORIES:
        if pos not in metrics:
            metrics[pos] = 0.0

    return metrics


# --- 3. Plotting Function ---

def plot_pos_analysis(df_pos_averages, feature_names, plot_title):
    """
    Generates a bar chart comparing the average percentage of specified features
    (POS tags or density) across all datasets, split by class.

    FIX: The bar color now uses a unique color for each dataset,
    allowing the legend to correctly differentiate them.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')

    class_types = ['Total', 'Positive', 'Negative']

    # Define a unique color map for each dataset
    dataset_names = df_pos_averages['Dataset'].unique()
    # Using 'tab10' for distinct colors (up to 10 datasets)
    # cmap = plt.cm.get_cmap('tab10')
    colors = sns.color_palette("Set2")  # , n_colors=len(unique_labels))
    dataset_color_map = {name: colors[i] for i, name in enumerate(dataset_names)}

    # Bar width and positions
    num_datasets = len(dataset_names)
    bar_width = 0.8 / num_datasets

    # Generate an X-axis position for each feature (e.g., Noun, Verb)
    x_pos = np.arange(len(feature_names))

    for i, class_type in enumerate(class_types):
        ax = axes[i]

        # Filter data for the current class type
        plot_data = df_pos_averages[df_pos_averages['Class'] == class_type]

        for j, dataset in enumerate(dataset_names):
            # Get the metrics for this dataset and class
            # Ensure the dataset exists in the data before plotting
            if not plot_data[plot_data['Dataset'] == dataset].empty:
                dataset_metrics = plot_data[plot_data['Dataset'] == dataset][feature_names].values[0]

                # Calculate the position for this dataset's bars
                x_offset = j * bar_width - (bar_width * (num_datasets - 1) / 2)

                # Set the relevant hatch
                hatches = {0: None, 1: 'x', 2: '.'}
                # Use the dataset-specific color
                ax.bar(x_pos + x_offset, dataset_metrics,
                       bar_width, label=dataset,
                       color=dataset_color_map[dataset],  # <-- FIXED: Use dataset's unique color
                       edgecolor='black', alpha=0.7, hatch=hatches[i])

        ax.set_title(f'{class_type} Samples Average')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=30, ha='right')
        ax.set_ylabel('Average Percentage of Total Words (%)')
        ax.grid(axis='y', alpha=0.5, linestyle='--')

    # Add a single legend (handles are correctly colored now)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Dataset')

    plt.show()


def plot_pos_heatmap(df_pos_averages):
    """
    Generates three heatmaps comparing the average percentage of POS categories
    across all datasets, split by class (Total, Positive, Negative).
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle('Heatmap of Average POS Frequencies (%) Across Datasets', fontsize=16, fontweight='bold')

    class_types = ['Total', 'Positive', 'Negative']

    # Features to include in the heatmap (all POS categories, excluding density)
    heatmap_features = POS_CATEGORIES
    dataset_names = df_pos_averages['Dataset'].unique()

    # Determine global max and min values for consistent color scaling
    vmax = df_pos_averages[heatmap_features].max().max()
    vmin = df_pos_averages[heatmap_features].min().min()

    for i, class_type in enumerate(class_types):
        ax = axes[i]

        # Filter data and pivot
        plot_data = df_pos_averages[df_pos_averages['Class'] == class_type]

        # Pivot table: Index=POS Categories (Rows), Columns=Datasets
        heatmap_matrix = plot_data.set_index('Dataset')[heatmap_features].T

        # Plot the heatmap using viridis map
        # Note: I'm using imshow which is standard matplotlib
        im = ax.imshow(heatmap_matrix.values, cmap='viridis', vmax=vmax, vmin=vmin)

        # Add labels/ticks
        ax.set_xticks(np.arange(len(dataset_names)))
        ax.set_yticks(np.arange(len(heatmap_features)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(heatmap_features)

        ax.set_title(f'{class_type} Samples')

        # Loop over data dimensions and create text annotations
        for y in range(heatmap_matrix.shape[0]):
            for x in range(heatmap_matrix.shape[1]):
                value = heatmap_matrix.iloc[y, x]
                # Determine text color based on background (simple heuristic: light/dark)
                text_color = "white" if value > vmax / 2 else "black"
                ax.text(x, y, f'{value:.1f}',
                        ha="center", va="center", color=text_color, fontsize=9)

    # Add a single color bar for the entire figure (based on the last plotted image)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Average Percentage of Total Words (%)')

    plt.show()

# --- 4. Main Execution Block ---

def main():
    # 1. Initialize Datasets
    datasets_info = {}
    dataset_codenames = ['amazon', 'dadjokes', 'headlines', 'one_liners']
    dataset_names = ['Amazon', 'Dad Jokes', 'Headlines', 'One Liners']

    for i, dataset in enumerate(dataset_codenames):
        data = load_dataset(dataset_name=dataset, test_percent=1125.0,
                            add_instruction=False, instruction_version=0, with_val=False,
                            train_size=5000)

        if dataset == 'amazon':
            print('*** Removing product names from Amazon dataset... ***')
            remove_product = lambda example: {'text': example['text'][example['text'].index(' [SEP] ') + len(' [SEP] '):]}
            for split in ['train', 'test']:
                data[split] = data[split].map(
                    remove_product,  # your function
                    batched=False  # or True if your fn accepts batches
                )

        datasets_info[dataset_names[i]] = data['train'].to_pandas()
    all_dataset_pos_averages = []

    # 2. Analyze each dataset
    for name, df in datasets_info.items():
        # Apply POS analysis to every sample
        pos_df = df['text'].apply(calculate_pos_metrics).apply(pd.Series)
        df = pd.concat([df, pos_df], axis=1)

        # Calculate Averages for Total, Positive, and Negative Classes

        # Total Average (All samples)
        total_avg = df.mean(numeric_only=True).to_dict()
        total_avg.update({'Dataset': name, 'Class': 'Total'})
        all_dataset_pos_averages.append(total_avg)

        # Positive Class Average
        positive_avg = df[df['label'] == 1].mean(numeric_only=True).to_dict()
        positive_avg.update({'Dataset': name, 'Class': 'Positive'})
        all_dataset_pos_averages.append(positive_avg)

        # Negative Class Average
        negative_avg = df[df['label'] == 0].mean(numeric_only=True).to_dict()
        negative_avg.update({'Dataset': name, 'Class': 'Negative'})
        all_dataset_pos_averages.append(negative_avg)

    # Convert results to a single DataFrame for plotting
    df_results = pd.DataFrame(all_dataset_pos_averages)

    # 3. Generate Plots

    # Plot 1: Major POS Categories
    pos_features = ['Noun', 'Verb', 'Adjective', 'Adverb']
    plot_pos_analysis(
        df_results,
        feature_names=pos_features,
        plot_title='Comparative POS Frequencies (Average % of Total Words)'
    )

    # Plot 2: Lexical Density (Analysis #2)
    plot_pos_analysis(
        df_results,
        feature_names=['Lexical Density'],
        plot_title='Comparative Lexical Density (Average % of Lexical Words)'
    )

    # Plot 3: Specific Ratios (e.g., Pronouns and Determiners)
    ratio_features = ['Pronoun', 'Determiner', 'Other']
    plot_pos_analysis(
        df_results,
        feature_names=ratio_features,
        plot_title='Comparative Function Word Frequencies (Average % of Total Words)'
    )

    # Plot 4: Heatmaps for POS Frequencies
    plot_pos_heatmap(df_results)


if __name__ == '__main__':
    # List of resources to ensure are present before running main()
    required_resources = ['punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']

    # Check and download if any standard resource is missing
    for resource in required_resources:
        try:
            # Check the resource path
            if resource == 'punkt':
                # Check for the main punkt resource
                nltk.data.find(f'tokenizers/{resource}')
            else:
                # Check for the tagger
                nltk.data.find(f'taggers/{resource}')
        except LookupError:
            print(f"Downloading required NLTK package: '{resource}'...")
            nltk.download(resource, quiet=True)

    # Specific fix for the 'punkt_tab' error you encountered.
    # This resource is sometimes needed explicitly, even if 'punkt' is downloaded.
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        print("Downloading required NLTK package: 'punkt_tab'...")
        nltk.download('punkt_tab', quiet=True)


    print("NLTK resource checks complete. Running analysis...")

    main()