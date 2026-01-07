import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASETS = ['amazon', 'dadjokes', 'headlines', 'one_liners']

llama_loo_median = {
    'amazon': [67, 87, 90, 86],
    'dadjokes': [83, 59, 91, 87],
    'headlines': [82, 88, 70, 84],
    'one_liners': [84, 88, 92, 66]
}

llama_loo_leftout = {
    'amazon': [68, 86, 90, 86],
    'dadjokes': [72, 70, 76, 76],
    'headlines': [81, 88, 70, 84],
    'one_liners': [77, 85, 81, 70]
}

mistral_loo_median = {
    'amazon': [61, 94, 96, 93],
    'dadjokes': [89, 54, 97, 94],
    'headlines': [90, 94, 71, 92],
    'one_liners': [88, 94, 96, 70]
}

mistral_loo_leftout = {
    'amazon': [67, 93, 92, 91],
    'dadjokes': [88, 55, 97, 93],
    'headlines': [89, 96, 73, 92],
    'one_liners': [88, 93, 96, 71]
}

models_result_dict = {
    'llama': {'median': llama_loo_median, 'leftout': llama_loo_leftout},
    'mistral': {'median': mistral_loo_median, 'leftout': mistral_loo_leftout}
}

# Convert list values to dict values
def convert_lists_to_dicts(data_dict):
    return {
        key: dict(zip(DATASETS, values))
        for key, values in data_dict.items()
    }

for model, results in models_result_dict.items():
    results_median = results['median']
    results_leftout = results['leftout']

    loo_median_dict = convert_lists_to_dicts(results_median)
    loo_leftout_dict = convert_lists_to_dicts(results_leftout)

    # Prepare data
    palette = sns.color_palette("Set2", n_colors=len(DATASETS))

    for i, dataset in enumerate(DATASETS):
        color = palette[i]

        # Point 1: from median dict
        x1 = loo_median_dict[dataset][dataset]
        y1 = np.mean([
            v for k, v in loo_median_dict[dataset].items() if k != dataset
        ])

        # Point 2: from leftout dict
        x2 = loo_leftout_dict[dataset][dataset]
        y2 = np.mean([
            v for k, v in loo_leftout_dict[dataset].items() if k != dataset
        ])

        # Plot line connecting the two points
        plt.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)

        # Scatter the two points
        plt.scatter(x1, y1, color=color, marker='o', s=100, label=f"{dataset} (median)")
        plt.scatter(x2, y2, color=color, marker='X', s=100, label=f"{dataset} (leftout)")

        # Annotate with value
        # plt.annotate(f"({x1}, {y1:.1f})", (x1, y1), textcoords="offset points", xytext=(-5, 0), ha='right')
        # plt.annotate(f"({x2}, {y2:.1f})", (x2, y2), textcoords="offset points", xytext=(5, 20), ha='left')



    plt.xlabel("Self Accuracy")
    plt.ylabel("Avg Transfer Accuracy")
    plt.title(f"Self vs Transfer Accuracy - {model}: Median vs Leftout")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
