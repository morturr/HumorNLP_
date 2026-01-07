import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import sys
import pandas as pd
import umap
sys.path.append('../')
from FlanT5.data_loader import (
    load_dataset,
)

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch


# 1️⃣  Load tokenizer & model (fp16 on GPU if available)
device      = "cuda" if torch.cuda.is_available() else "cpu"
# model_name  = "mistralai/Mistral-7B-v0.1"      # or any other Mistral checkpoint
# model_name_short = 'Mistral'
model_name  = "meta-llama/Llama-2-7b-hf"      # or any other LLAMA checkpoint
model_name_short = 'LLaMA-2'

tokenizer   = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else None)
model.to(device)
model.eval()

def encode(texts, batch_size=8, max_len=1024):
    """Return a matrix [n_texts × hidden_size] of mean-pooled embeddings."""
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch     = texts[i:i+batch_size]
            encoded   = tokenizer(batch,
                                  padding=True,
                                  truncation=True,
                                  max_length=max_len,
                                  return_tensors="pt").to(device)
            outputs   = model(**encoded).last_hidden_state       # [B, L, H]
            mask      = encoded.attention_mask.unsqueeze(-1)     # [B, L, 1]
            summed    = (outputs * mask).sum(dim=1)              # [B, H]
            counts    = mask.sum(dim=1)                          # [B, 1]
            mean_pool = summed / counts                          # [B, H]
            all_embs.append(mean_pool.cpu())
    return torch.cat(all_embs, dim=0).numpy()

def average_pairwise_cosine(embeddings):
    """
    embeddings: ndarray shape [N, D]
    returns the mean cosine similarity over all pairs i<j
    """
    sims   = cosine_similarity(embeddings)      # full N×N matrix
    tril_i = np.tril_indices_from(sims, k=-1)   # indices below main diagonal
    return sims[tril_i].mean()

# ----------  Cross‑dataset similarity ----------
def average_cross_cosine(emb_A, emb_B):
    """
    emb_A: ndarray [N,H]  (dataset‑1 embeddings)
    emb_B: ndarray [M,H]  (dataset‑2 embeddings)
    Returns mean cosine similarity over all N×M pairs.
    """
    # sklearn’s cosine_similarity can take two matrices and returns N×M
    sims   = cosine_similarity(emb_A, emb_B)
    return sims.mean()

def average_cosine(A, B):
    sims = cosine_similarity(A, B)
    if A is B:
        # Same-dataset case: use only lower triangle, excluding diagonal
        tril = np.tril_indices_from(sims, k=-1)
        return sims[tril].mean()
    else:
        # Different datasets: use full matrix
        return sims.mean()


def average_nearest_cross_cosine(emb_A, emb_B, class_label):
    """
    emb_A: ndarray [N_A, H] (embeddings of the source class, e.g., Positive)
    emb_B: ndarray [N_B, H] (embeddings of the target class, e.g., Negative)
    class_label: str ('Pos->Neg' or 'Neg->Pos') for descriptive output

    For each sample in emb_A, finds the nearest neighbor in emb_B (by max cosine sim),
    and returns the average of these max similarities. Returns NaN if a class is empty.
    """
    if len(emb_A) == 0 or len(emb_B) == 0:
        print(f"Warning: Skipping {class_label} calculation due to empty class.")
        return np.nan

    # 1. Compute all pairwise cosine similarities between A and B (N_A x N_B matrix)
    sims = cosine_similarity(emb_A, emb_B)  # [N_A, N_B]

    # 2. For each row (sample in A), find the maximum similarity (its nearest B neighbor)
    max_sims = sims.max(axis=1)  # [N_A] vector of max similarities

    # 3. Return the average of these maximum similarities
    return max_sims.mean()


def compute_nearest_neighbor_matrix(datasets_dict, dataset_embeddings_dict, DATASETS, split):
    """
    Computes and prints a table of within-dataset nearest-neighbor cosine similarities.
    For each dataset, it calculates: Avg(Pos -> NN_Neg) and Avg(Neg -> NN_Pos).
    """
    print("\n" + "=" * 50)
    print("🤖 Computing Within-Dataset Nearest Neighbor Similarity 🤖")
    print("=" * 50)

    # Initialize a new DataFrame just for these two metrics
    metrics = ['Pos->Neg', 'Neg->Pos']
    nn_matrix = pd.DataFrame(index=DATASETS, columns=metrics)

    for dataset_name in DATASETS:
        print(f"Processing dataset: {dataset_name}...")

        # Get all embeddings and labels for the current dataset
        embeddings = dataset_embeddings_dict[dataset_name]
        labels = np.array(datasets_dict[dataset_name][split]['label'])

        # Filter embeddings by class label (1 is positive, 0 is negative)
        pos_mask = labels == 1
        neg_mask = labels == 0

        emb_pos = embeddings[pos_mask]
        emb_neg = embeddings[neg_mask]

        # Calculate Pos -> Neg Nearest Neighbor Similarity
        sim_pos_to_neg = average_nearest_cross_cosine(emb_pos, emb_neg, f'{dataset_name} Pos->Neg')
        nn_matrix.loc[dataset_name, 'Pos->Neg'] = sim_pos_to_neg

        # Calculate Neg -> Pos Nearest Neighbor Similarity
        sim_neg_to_pos = average_nearest_cross_cosine(emb_neg, emb_pos, f'{dataset_name} Neg->Pos')
        nn_matrix.loc[dataset_name, 'Neg->Pos'] = sim_neg_to_pos

    nn_matrix = nn_matrix.astype(float)
    nn_matrix.to_csv(f"{model_name_short}_nearest_neighbor_similarity_{split}.csv", float_format="%.4f")

    print(f"\nSaved Nearest Neighbor Similarity to CSV. Results for {split}:")
    print(nn_matrix)

    # 2. Visualize as Custom Bar Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Custom Plotting Setup ---
    set2_colors = sns.color_palette("Set2", n_colors=len(DATASETS))
    # Note: Based on your second image, it seems the two directions have different hatch styles:
    HATCH_POS_NEG = 'x'  # Hatch for Pos->Neg (Appears to be cross-hatch in your plot)
    HATCH_NEG_POS = '.'  # Hatch for Neg->Pos (Appears to be a dense dot pattern in your plot)

    # Define bar width and gap
    bar_width = 0.35
    gap = 0.05
    x = np.arange(len(DATASETS))

    # Plotting 'Pos->Neg Nearest Neighbor Sim'
    pos_bars = ax.bar(x - (bar_width / 2 + gap / 2), nn_matrix['Pos->Neg'],
                      bar_width, label='Pos->Neg',
                      align='center', zorder=2)

    # Plotting 'Neg->Pos Nearest Neighbor Sim'
    neg_bars = ax.bar(x + (bar_width / 2 + gap / 2), nn_matrix['Neg->Pos'],
                      bar_width, label='Neg->Pos',
                      align='center', zorder=2)

    # --- Apply custom color and hatch to each bar ---
    for i in range(len(DATASETS)):
        color = set2_colors[i]

        # Pos->Neg Bar
        pos_bars[i].set_color(color)
        pos_bars[i].set_edgecolor('black')
        pos_bars[i].set_hatch(HATCH_POS_NEG)

        # Neg->Pos Bar
        neg_bars[i].set_color(color)
        neg_bars[i].set_edgecolor('black')
        neg_bars[i].set_hatch(HATCH_NEG_POS)

        # Set X ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(['Amazon', 'Dad Jokes', 'Headlines', 'One Liners'])

    # Add data labels on top of the bars
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold', rotation=0)

    add_bar_labels(pos_bars)
    add_bar_labels(neg_bars)

    # --- MANUAL LEGEND CREATION ---
    # We use matplotlib.patches.Patch to create custom legend handles.

    # 1. Handle for Pos->Neg
    pos_neg_handle = Patch(
        facecolor='white',  # Set fill color to white
        edgecolor='black',  # Keep a black border
        hatch=HATCH_POS_NEG,  # Use the defined hatch pattern
        label='Pos->Neg'
    )

    # 2. Handle for Neg->Pos
    neg_pos_handle = Patch(
        facecolor='white',  # Set fill color to white
        edgecolor='black',  # Keep a black border
        hatch=HATCH_NEG_POS,  # Use the defined hatch pattern
        label='Neg->Pos'
    )

    # Apply the custom legend handles and labels
    ax.legend(
        handles=[pos_neg_handle, neg_pos_handle],
        labels=[pos_neg_handle.get_label(), neg_pos_handle.get_label()],
        title='Direction'
    )

    # Final plot settings
    plt.title(f"{model_name_short}: Within-Dataset Nearest Neighbor Cosine Similarity")
    plt.ylabel("Average Cosine Similarity")
    plt.xlabel("Dataset")

    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    max_val = nn_matrix.values.max()
    plt.ylim(0, max_val * 1.15 if not np.isnan(max_val) else 1)

    plt.tight_layout()
    plt.show()

    return nn_matrix

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # df_similarity_train_mistral = pd.read_csv(f"mistral_amazon_only_questions_dataset_pairwise_cosine_similarity_train.csv", index_col=0)
    # df_similarity_test_mistral = pd.read_csv(f"mistral_amazon_only_questions_dataset_pairwise_cosine_similarity_test.csv", index_col=0)
    # df_similarity_train_llama = pd.read_csv(f"llama_amazon_only_questions_dataset_pairwise_cosine_similarity_train.csv", index_col=0)
    # df_similarity_test_llama = pd.read_csv(f"llama_amazon_only_questions_dataset_pairwise_cosine_similarity_test.csv", index_col=0)

    DATASETS = ['amazon', 'dadjokes', 'headlines', 'one_liners']
    datasets_dict = {}

    for dataset_name in DATASETS:
        data = load_dataset(dataset_name=dataset_name, test_percent=1125.0,
                            add_instruction=False, instruction_version=0, with_val=False,
                            train_size=5000)

        if dataset_name == 'amazon':
            print('*** Removing product names from Amazon dataset... ***')
            remove_product = lambda example: {'text': example['text'][example['text'].index(' [SEP] ') + len(' [SEP] '):]}
            for split in ['train', 'test']:
                data[split] = data[split].map(
                    remove_product,  # your function
                    batched=False  # or True if your fn accepts batches
                )

        datasets_dict[dataset_name] = data

    # ---------- Compute Embeddings ----------
    all_embeddings = []
    # all_labels = []
    dataset_labels = []
    class_labels = []

    # for split in ['train', 'test']:
    for split in ['train']:
        print(f"Computing embeddings for split: {split}, length: {len(datasets_dict['amazon'][split]['text'])}")
        dataset_embeddings_dict = {name: encode(texts[split]['text']) for name, texts in datasets_dict.items()}

        APPLY_PCA_TSNE_UMAP = False
        TSNE_PCA_UMAP = 'tsne'  # 'pca' or 'tsne' or 'umap
        for TSNE_PCA_UMAP in ['tsne', 'umap']:
            if APPLY_PCA_TSNE_UMAP:
                # --- 1. Gather ALL Embeddings and Labels ---
                all_embeddings = []
                dataset_labels = []  # Dataset name (for coloring in all-pos/all-neg plots)
                class_labels = []  # Class label (0 or 1 for coloring in dataset plots)

                for name, embeddings in dataset_embeddings_dict.items():
                    labels = datasets_dict[name][split]['label']
                    all_embeddings.extend(embeddings)
                    dataset_labels.extend([name] * len(embeddings))
                    class_labels.extend(labels)

                X = np.vstack(all_embeddings)  # shape [N_total, D]
                class_labels_np = np.array(class_labels)
                dataset_labels_np = np.array(dataset_labels)

                # --- 2. Define Scenarios for Dimensionality Reduction ---
                SCENARIOS = []

                #Scenario 0: All Data (Color by Dataset)
                all_indices = np.full(len(class_labels), True)
                SCENARIOS.append(('All Data', all_indices))

                # Scenario A: All Positive Labels (Color by Dataset)
                pos_indices = class_labels_np == 1
                SCENARIOS.append(('All Positive Labels', pos_indices))

                # Scenario B: All Negative Labels (Color by Dataset)
                neg_indices = class_labels_np == 0
                SCENARIOS.append(('All Negative Labels', neg_indices))

                # Scenario C: Each Dataset Separately (Color by Class)
                for i, dataset_name in enumerate(DATASETS):
                    # Indices for the current dataset (assuming data is stacked in DATASETS order)
                    dataset_indices = dataset_labels_np == dataset_name
                    SCENARIOS.append((f'{dataset_name} (Pos & Neg)', dataset_indices))

                ALGO_NAME_MAP = {'tsne': 't-SNE', 'umap': 'UMAP', 'pca': 'PCA'}
                algo_name = ALGO_NAME_MAP[TSNE_PCA_UMAP]

                # --- 3. Iterate, Reduce, and Plot ---
                for scenario_name, indices_to_select in SCENARIOS:
                    # 3.1 Select the subset of embeddings
                    X_subset = X[indices_to_select]
                    subset_class_labels = class_labels_np[indices_to_select]
                    subset_dataset_labels = dataset_labels_np[indices_to_select]

                    if len(X_subset) == 0:
                        print(f"Skipping '{scenario_name}': No samples in this subset.")
                        continue

                    N = len(X_subset)
                    shuffled_indices = np.arange(N)
                    np.random.RandomState(42).shuffle(shuffled_indices)

                    # Apply the shuffled indices to the data and ALL corresponding label arrays
                    subset_class_labels_shuffled = subset_class_labels[shuffled_indices]
                    subset_dataset_labels_shuffled = subset_dataset_labels[shuffled_indices]

                    # Choose if shuffle is needed for plotting
                    X_to_fit = X_subset
                    # X_to_fit = X_subset[shuffled_indices]

                    # 3.2 Apply Dimensionality Reduction to the subset
                    print(f"Applying {algo_name} to: {scenario_name} ({len(X_to_fit)} samples)")

                    if TSNE_PCA_UMAP == 'tsne':
                        # Use default n_jobs (or n_jobs=-1 if available)
                        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
                        X_reduced = tsne.fit_transform(X_to_fit)
                    elif TSNE_PCA_UMAP == 'umap':
                        umap_model = umap.UMAP(n_components=2, random_state=42)
                        X_reduced = umap_model.fit_transform(X_to_fit)
                    else:  # pca
                        pca = PCA(n_components=2)
                        X_reduced = pca.fit_transform(X_to_fit)

                    # 3.3 Prepare labels for the selected subset
                    # Choose if shuffle is needed for plotting
                    # subset_class_labels_for_plot = subset_class_labels_shuffled
                    # subset_dataset_labels_for_plot = subset_dataset_labels_shuffled
                    subset_class_labels_for_plot = subset_class_labels
                    subset_dataset_labels_for_plot = subset_dataset_labels

                    # 3.4 Plotting
                    plt.figure(figsize=(8, 6))

                    if scenario_name in ['All Data', 'All Positive Labels', 'All Negative Labels']:
                        # Plotting A & B: Color by Dataset
                        unique_datasets = np.unique(subset_dataset_labels_for_plot)
                        colors = sns.color_palette("Set2", n_colors=len(unique_datasets))

                        # Define dict for display names
                        dataset_display_names = {'amazon': 'Amazon', 'dadjokes': 'Dad Jokes',
                                                 'headlines': 'Headlines', 'one_liners': 'One Liners'}

                        for i, dataset in enumerate(unique_datasets):
                            dataset_mask = subset_dataset_labels_for_plot == dataset
                            plt.scatter(X_reduced[dataset_mask, 0], X_reduced[dataset_mask, 1],
                                        label=dataset_display_names[dataset], color=colors[i], alpha=0.8, s=7, marker='o')

                        plot_title = f"{algo_name} of {model_name_short} Embeddings - {scenario_name}"

                    else:  # Plotting C: Individual dataset plots (Color by Class)
                        positive_mask = subset_class_labels_for_plot == 1
                        negative_mask = subset_class_labels_for_plot == 0

                        plt.scatter(X_reduced[positive_mask, 0], X_reduced[positive_mask, 1],
                                    label='Positive', color='green', alpha=0.7, s=7, marker='o')

                        plt.scatter(X_reduced[negative_mask, 0], X_reduced[negative_mask, 1],
                                    label='Negative', color='red', alpha=0.5, s=7, marker='x')

                        # Extract the dataset name from the scenario_name
                        dataset_name_for_title = dataset_display_names[scenario_name.split(' ')[0]]
                        plot_title = f"{algo_name} of {model_name_short} Embeddings - {dataset_name_for_title} (Pos & Neg)"

                    plt.xlabel(f"{algo_name} Dim 1")
                    plt.ylabel(f"{algo_name} Dim 2")
                    plt.title(plot_title)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

        COMPUTE_SIMILARITY_MATRIX = False
        if COMPUTE_SIMILARITY_MATRIX:
            # ---------- Pairwise Similarity Matrix ----------
            matrix = pd.DataFrame(index=DATASETS, columns=DATASETS)

            for i, name1 in enumerate(DATASETS):
                for j, name2 in enumerate(DATASETS):
                    if pd.isna(matrix.loc[name1, name2]):
                        sim = average_cosine(dataset_embeddings_dict[name1], dataset_embeddings_dict[name2])
                        matrix.loc[name1, name2] = sim
                        matrix.loc[name2, name1] = sim  # symmetry

            matrix = matrix.astype(float)

            # ---------- Save to CSV ----------
            matrix.to_csv(f"dataset_pairwise_cosine_similarity_{split}.csv", float_format="%.4f")

            print(f"Saved pairwise similarity matrix for {split}:")
            print(matrix)

        COMPUTE_NN_MATRIX = True
        if COMPUTE_NN_MATRIX:
            # ⚠️ This block computes the nearest neighbor metrics, entirely separate
            # from the cross-dataset similarity matrix.
            nn_similarity_matrix = compute_nearest_neighbor_matrix(
                datasets_dict,
                dataset_embeddings_dict,
                DATASETS,
                split
            )



