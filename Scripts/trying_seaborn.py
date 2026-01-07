import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

single_accuracy_NEW = {'LLaMA Amazon': [88.23],
                         'Mistral Amazon': [90.84],
                         'LLaMA Dad Jokes': [92.79],
                         'Mistral Dad Jokes': [94.04],
                         'LLaMA Headlines': [90.21],
                         'Mistral Headlines': [96.69],
                         'LLaMA One Liners': [86.77],
                         'Mistral One Liners': [94.67],
                         }

pair_accuracy_NEW = {'LLaMA Amazon': [82.47, 82.08, 86.13],
                       'Mistral Amazon': [89.48, 89.33, 90.24],
                       'LLaMA Dad Jokes': [89.53, 89.42, 89.71],
                       'Mistral Dad Jokes': [94.98, 90.51, 95.44],
                       'LLaMA Headlines': [89.5, 91.69, 92.78],
                       'Mistral Headlines': [96.47, 94.67, 96.73],
                       'LLaMA One Liners': [91.24, 92.02, 91.33],
                       'Mistral One Liners': [94.4, 94.04, 94.36],
                       }

loo_accuracy_NEW = {'LLaMA Amazon': [83.89, 86.04, 84.04],
                      'Mistral Amazon': [89.8, 89.53, 89.87],
                      'LLaMA Dad Jokes': [87.82, 88.71, 89.33],
                      'Mistral Dad Jokes': [94.33, 94.8, 95.6],
                      'LLaMA Headlines': [89.44, 89.78, 90.69],
                      'Mistral Headlines': [95.89, 96.42, 96.04],
                      'LLaMA One Liners': [88.09, 87.24, 87.87],
                      'Mistral One Liners': [93.56, 93.51, 93.44],
                      }

single_transfer_NEW = {'LLaMA Amazon': [62.63, 58.07, 62.33],
                         'Mistral Amazon': [68.82, 65.04, 63.67],
                         'LLaMA Dad Jokes': [65.15, 56.78, 61.84],
                         'Mistral Dad Jokes': [61.91, 55.75, 51.16],
                         'LLaMA Headlines': [65.39, 59.32, 54.0],
                         'Mistral Headlines': [74.51, 68.11, 67.38],
                         'LLaMA One Liners': [61.11, 70.09, 65.43],
                         'Mistral One Liners': [71.87, 70.91, 61.53],
                         }

pair_transfer_NEW = {'LLaMA Amazon': [62.71, 73.8, 63.26],
                       'Mistral Amazon': [70.5, 66.26, 67.58],
                       'LLaMA Dad Jokes': [61.97, 54.9, 55.59],
                       'Mistral Dad Jokes': [66.93, 52.4, 53.42],
                       'LLaMA Headlines': [67.04, 65.07, 65.29],
                       'Mistral Headlines': [73.5, 70.09, 74.07],
                       'LLaMA One Liners': [69.26, 68.6, 70.83],
                       'Mistral One Liners': [73.94, 66.6, 69.64],
                       }

loo_transfer_NEW = {'LLaMA Amazon': [68.93],
                      'Mistral Amazon': [65.51],
                      'LLaMA Dad Jokes': [57.18],
                      'Mistral Dad Jokes': [55.24],
                      'LLaMA Headlines': [68.13],
                      'Mistral Headlines': [73.38],
                      'LLaMA One Liners': [69.35],
                      'Mistral One Liners': [73.6],
                      }

export_plot_type = 'Transfer'
# export_plot_type = 'Self'

if export_plot_type == 'Transfer':
    single_data = single_transfer_NEW
    pair_data = pair_transfer_NEW
    loo_data = loo_transfer_NEW
elif export_plot_type == 'Self':
    single_data = single_accuracy_NEW
    pair_data = pair_accuracy_NEW
    loo_data = loo_accuracy_NEW
else:
    raise ValueError("Invalid export_plot_type. Choose 'Transfer' or 'Self'.")

def get_averages(data):
    return {key: np.average(values) for key, values in data.items()}

def get_differences(data1, data2, data3):
    return {key: (data2[key] - data1[key], data3[key] - data2[key], data3[key] - data1[key]) for key in data1.keys()}

# PRINT_DIFFS = True
PRINT_DIFFS = False

if PRINT_DIFFS:
    single_avg, pair_avg, loo_avg = (
        get_averages(single_data), get_averages(pair_data), get_averages(loo_data))

    accuracy_diff = get_differences(single_avg, pair_avg, loo_avg)
    for i in range(3):
        if i == 0:
            diff_type = 'Pair to Single'
        elif i == 1:
            diff_type = 'Leave One Out to Pair'
        elif i == 2:
            diff_type = 'Leave One Out to Single'

        for model_name in ['LLaMA', 'Mistral']:
            diffs_avg = np.average([value[i] for key, value in accuracy_diff.items() if model_name in key])
            print(f'Difference of {diff_type}, {model_name}: {diffs_avg:.2f}')
            for key, value in accuracy_diff.items():
                if model_name in key:
                    print(f'    {key}: {value[i]:.2f}')



# CREATE_PLOTS = False
CREATE_PLOTS = True
if CREATE_PLOTS:
    # Calculate means and std
    def get_mean_std(data):
        return np.mean(data), np.std(data)

    def get_max(data):
        return np.max(data)


    # Create a DataFrame for plotting
    categories = list(single_transfer_NEW.keys())

    # Prepare the data
    means = {'single': [], 'pair': [], 'loo': []}
    stds = {'single': [], 'pair': [], 'loo': []}
    maxs = {'single': [], 'pair': [], 'loo': []}


    for category in categories:
        mean_single, std_single = get_mean_std(single_data[category])
        mean_pair, std_pair = get_mean_std(pair_data[category])
        mean_loo, std_loo = get_mean_std(loo_data[category])

        max_single = get_max(single_data[category])
        max_pair = get_max(pair_data[category])
        max_loo = get_max(loo_data[category])

        means['single'].append(mean_single)
        means['pair'].append(mean_pair)
        means['loo'].append(mean_loo)

        stds['single'].append(std_single)
        stds['pair'].append(std_pair)
        stds['loo'].append(std_loo)

        maxs['single'].append(max_single)
        maxs['pair'].append(max_pair)
        maxs['loo'].append(max_loo)

    # Convert to DataFrame
    df = pd.DataFrame({
        'Category': categories,
        'Single': means['single'],
        'Pair': means['pair'],
        'Loo': means['loo'],
        'Single_std': stds['single'],
        'Pair_std': stds['pair'],
        'Loo_std': stds['loo'],
        'Single_max': maxs['single'],
        'Pair_max': maxs['pair'],
        'Loo_max': maxs['loo']
    })

    # Colors for categories
    colors = sns.color_palette("Set2", n_colors=5)  # 5 distinct colors

    # Create a plot with multiple lines and error bars
    plt.figure(figsize=(7, 6))

    # Plot each line with error bars for each dataset
    for idx, category in enumerate(categories):
        # Get the means and std for the current category
        if (idx // 2) != 3:
            continue
        single_mean = df.loc[idx, 'Single']
        pair_mean = df.loc[idx, 'Pair']
        loo_mean = df.loc[idx, 'Loo']

        single_std = df.loc[idx, 'Single_std']
        pair_std = df.loc[idx, 'Pair_std']
        loo_std = df.loc[idx, 'Loo_std']

        single_max = df.loc[idx, 'Single_max']
        pair_max = df.loc[idx, 'Pair_max']
        loo_max = df.loc[idx, 'Loo_max']

        # Use different colors for each dataset
        color_idx = idx // 2
        color = colors[color_idx]

        # Choose dotted lines or regular lines based on the category
        linestyle = '--' if 'LLaMA' in category else '-'
        marker = '^' if 'LLaMA' in category else 'o'
        x_axis_captions = ['Single Training', 'Pair Training', 'Triple Training'] if export_plot_type == 'Transfer' \
                        else ['Single Training\n(5000 in-domain samples)',
                              'Pair Training\n(2500 in-domain samples)',
                              'Triple Training\n(1666 in-domain samples)']

        # Add lines connecting the dots of the same category
        plt.plot(x_axis_captions, [single_mean, pair_mean, loo_mean],
                 color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=9)

        # plt.plot(x_axis_captions, [single_max, pair_max, loo_max],
        #          color=color, marker=marker, linestyle=linestyle, linewidth=2, markersize=9)
        plt.xticks(fontsize=13)

    # Customize the plot
    plt.title(f'Mean {export_plot_type} Accuracy for Each Dataset and Model',
              fontsize=14)#, fontweight='bold')
    plt.ylabel(f'Mean {export_plot_type} Accuracy', fontsize=14)#, fontweight= 'bold')
    # plt.title(f'Max {export_plot_type} Accuracy for Each Dataset and Model', fontsize=17, fontweight='bold')
    # plt.ylabel(f'Max {export_plot_type} Accuracy', fontsize=17, fontweight= 'bold')

    plt.xlabel('Experiment Type', fontsize=14)#, fontweight= 'bold')
    if export_plot_type == 'Transfer':
        plt.ylim(54, 75)  # Control the y-axis range to better visualize the data
    else:
        plt.ylim(82, 98)

    # Add a legend
    # plt.legend(categories, loc='upper left', bbox_to_anchor=(0.07, 1), ncol=1, fontsize=12) # for legend on top of the plot
    # plt.legend(categories, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left') # for legend next to the plot
    plt.tight_layout()

    # Show the plot
    plt.show()
