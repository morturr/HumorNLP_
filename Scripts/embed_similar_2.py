# --- Original Code: Stack All Embeddings ---
# Note: You still need to gather all embeddings first to have them ready for slicing
all_embeddings = []
dataset_labels = []  # Dataset name (for color)
class_labels = []  # Class label (for shape)

for name, embeddings in dataset_embeddings_dict.items():
    labels = datasets_dict[name][split]['label']
    all_embeddings.extend(embeddings)
    dataset_labels.extend([name] * len(embeddings))
    class_labels.extend(labels)

X = np.vstack(all_embeddings)  # shape [N, D]
class_labels_np = np.array(class_labels)

# -------------------------------------------------------------
# Define the scenarios you want to visualize
# -------------------------------------------------------------

# This is a list of tuples: (scenario_name, indices_to_select)
# indices_to_select is a boolean mask or index array
SCENARIOS = []

# 1. All Positive Labels
pos_indices = class_labels_np == 1
SCENARIOS.append(('All Positive Labels', pos_indices))

# 2. All Negative Labels
neg_indices = class_labels_np == 0
SCENARIOS.append(('All Negative Labels', neg_indices))

# 3. Each Dataset Separately (Positive & Negative)
# Assuming you want to plot each dataset's points against *only* that dataset's
# reduced space (e.g., t-SNE trained just on Amazon data).
N_samples_per_dataset = 5000  # Based on your plotting logic
for i, dataset_name in enumerate(DATASETS):
    # Indices for the current dataset (assuming data is stacked in DATASETS order)
    dataset_start_idx = i * N_samples_per_dataset
    dataset_end_idx = (i + 1) * N_samples_per_dataset
    dataset_indices = np.arange(dataset_start_idx, dataset_end_idx)
    SCENARIOS.append((f'{dataset_name} (Pos & Neg)', dataset_indices))

ALGO_NAME_MAP = {'tsne': 't-SNE', 'umap': 'UMAP', 'pca': 'PCA'}
algo_name = ALGO_NAME_MAP[TSNE_PCA_UMAP]

for scenario_name, indices_to_select in SCENARIOS:
    # 1. Select the subset of embeddings
    X_subset = X[indices_to_select]

    if len(X_subset) == 0:
        print(f"Skipping '{scenario_name}': No samples in this subset.")
        continue

    # 2. Apply Dimensionality Reduction to the subset
    print(f"Applying {algo_name} to: {scenario_name} ({len(X_subset)} samples)")

    if TSNE_PCA_UMAP == 'tsne':
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_jobs=-1)
        X_reduced = tsne.fit_transform(X_subset)
    elif TSNE_PCA_UMAP == 'umap':
        umap_model = umap.UMAP(n_components=2, random_state=42)
        X_reduced = umap_model.fit_transform(X_subset)
    else:  # pca
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_subset)

    # 3. Prepare labels for the selected subset
    subset_class_labels = class_labels_np[indices_to_select]
    subset_dataset_labels = np.array(dataset_labels)[indices_to_select]

    # 4. Plotting
    plt.figure(figsize=(8, 6))

    # Determine plotting style based on scenario
    if scenario_name in ['All Positive Labels', 'All Negative Labels']:
        # For all-pos or all-neg plots, color by dataset
        unique_datasets = np.unique(subset_dataset_labels)
        colors = sns.color_palette("tab10", n_colors=len(unique_datasets))

        for i, dataset in enumerate(unique_datasets):
            dataset_mask = subset_dataset_labels == dataset
            plt.scatter(X_reduced[dataset_mask, 0], X_reduced[dataset_mask, 1],
                        label=dataset, color=colors[i], alpha=0.8, s=7, marker='o')

        plot_title = f"{algo_name} of {model_name_short} Embeddings - {scenario_name}"

    else:  # Individual dataset plots (Pos & Neg together)
        # For individual dataset plots, color by class (Pos/Neg)
        positive_mask = subset_class_labels == 1
        negative_mask = subset_class_labels == 0

        plt.scatter(X_reduced[positive_mask, 0], X_reduced[positive_mask, 1],
                    label='Positive', color='green', alpha=0.8, s=7, marker='o')

        plt.scatter(X_reduced[negative_mask, 0], X_reduced[negative_mask, 1],
                    label='Negative', color='red', alpha=0.8, s=7, marker='x')

        plot_title = f"{algo_name} of {model_name_short} Embeddings - {dataset_name} (Pos & Neg)"

    plt.xlabel(f"{algo_name} Dim 1")
    plt.ylabel(f"{algo_name} Dim 2")
    plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()