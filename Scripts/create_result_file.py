import pandas as pd
import os

USE_MODEL = 'Llama-2'
PATH = f'../{USE_MODEL}/Results'

DATASETS = ['amazon', 'dadjokes', 'headlines', 'one_liners', 'yelp_reviews']
# DATASETS = ['amazon']
PATH_TEMPLATE = PATH + '/{dataset}-models/'
TABLES_PATH = PATH + '/Tables/'

LOO_DATASETS = [f'loo_{dataset}' for dataset in DATASETS]
LOO_PATH = PATH + '/leave-one-out/'

# LOO_EVAL = False
LOO_EVAL = False

TRAIN_DATASETS = LOO_DATASETS if LOO_EVAL else DATASETS

METRICS = ['accuracy', 'f1', 'recall', 'precision']

PARAMS_COLUMN_NAMES = ['seed', 'learning_rate', 'per_device_train_batch_size',
                       'per_device_eval_batch_size', 'max_steps', 'lora_rank', 'lora_alpha', ]

# base_model = 'flan-t5-base'
BASE_MODEL = 'llama-2-7b'


# base_model = 'mistral-7b'


def load_score_table_from_files():
    # Create the score table for regular dataset
    if not LOO_EVAL:
        scores_df = pd.DataFrame()
        for dataset_name in TRAIN_DATASETS:
            root_dir = PATH_TEMPLATE.format(dataset=dataset_name)
            for _, dirs, _ in os.walk(root_dir):
                for dir_name in dirs:
                    inner_dir = os.path.join(root_dir, dir_name)
                    score_file_path = os.path.join(inner_dir, f'{dataset_name}_scores.csv')
                    df = pd.read_csv(score_file_path)
                    # for the first df, set its columns to be the columns of the overall dataframe (scores_df)
                    if scores_df.empty:
                        scores_df = df
                    else:
                        scores_df = pd.concat([scores_df, df])

    return scores_df


def get_top_accuracies(accuracies, get_top=3, get_by='median'):
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1][get_by], reverse=True)
    return sorted_accuracies[:get_top]


def find_best_accuracies(df, trained_dataset_name, which_best='All'):
    accuracies = {}
    best_trained, best_median, best_mean = (-1, None), (-1, None), (-1, None)

    for params, rows in df.groupby(by=PARAMS_COLUMN_NAMES):
        row_trained = rows[rows['evaluate_dataset'] == trained_dataset_name]
        row_others = rows[rows['evaluate_dataset'] != trained_dataset_name]
        curr_trained_accuracy = row_trained['accuracy'].values[0]
        others_accuracy_median = row_others['accuracy'].median()
        others_accuracy_mean = row_others['accuracy'].mean()

        best_trained = (curr_trained_accuracy, params) if curr_trained_accuracy > best_trained[0] else best_trained
        best_median = (others_accuracy_median, params) if others_accuracy_median > best_median[0] else best_median
        best_mean = (others_accuracy_mean, params) if others_accuracy_mean > best_mean[0] else best_mean

        curr_accs = {'trained': curr_trained_accuracy,
                     'median': others_accuracy_median,
                     'mean': others_accuracy_mean}

        accuracies[params] = curr_accs

    # print(f'Accuracies for dataset: {trained_dataset_name}')
    # print(f'Best trained accuracy = {best_trained}')
    # print(f'Best median accuracy = {best_median}')
    # print(f'Best mean accuracy = {best_mean}')

    if which_best == 'All':
        return accuracies, {'best_trained': best_trained,
                            'best_median': best_median,
                            'best_mean': best_mean}, accuracies

    # Return only best median
    if which_best == 'Median':
        return accuracies, {'best_median': best_median}, accuracies


def init_result_param_df():
    df = pd.read_excel(TABLES_PATH + 'result_param_metric_template.xlsx')
    df.fillna(method='ffill', axis=0, inplace=True)
    df.set_index(['metric', 'model', 'trained on', 'param_metric'], inplace=True)

    return df


def get_split_avg_single_df(df):
    # Iterate over the existing parameters combinations and create df of the average cv splits results
    df_avg_splits = pd.DataFrame(columns=df.columns)
    for params, rows in df.groupby(by=PARAMS_COLUMN_NAMES):
        for dataset_name in DATASETS:
            rows_of_dataset = rows[rows['evaluate_dataset'] == dataset_name]

            # Columns you want to average
            columns_to_average = ['accuracy', 'precision', 'recall', 'f1']

            # Calculate the mean for the relevant columns
            mean_values = rows_of_dataset[columns_to_average].mean()

            # Prepare a new row with average values and other relevant info
            # For example, 'split' can be labeled as 'average'
            new_row = rows_of_dataset.iloc[0].copy(deep=True)

            new_row.update(mean_values)
            new_row.update({'split_num': 'average'})
            new_row_df = pd.DataFrame([new_row], columns=df.columns)

            if df_avg_splits.empty:
                df_avg_splits = new_row_df
            else:
                df_avg_splits = pd.concat([df_avg_splits, new_row_df], axis=0)

    return df_avg_splits


def get_split_avg_df_all(df):
    df_avg_splits_all = pd.DataFrame()
    for dataset_name in TRAIN_DATASETS:
        df_curr_trained = df[(df['train_dataset'] == dataset_name)]
        # params_vals = {param_name: list(df_curr_trained[param_name].unique()) for param_name in PARAMS_COLUMN_NAMES}
        df_curr_trained.set_index(PARAMS_COLUMN_NAMES, inplace=True)

        df_curr_trained_avg_splits = get_split_avg_single_df(df_curr_trained)
        df_curr_trained_avg_splits.index.names = df_curr_trained.index.names

        if df_avg_splits_all.empty:
            df_avg_splits_all = df_curr_trained_avg_splits

        else:
            df_avg_splits_all = pd.concat([df_curr_trained_avg_splits, df_avg_splits_all])

    return df_avg_splits_all


def save_result_param_df(result_param_df, top_params_df):
    from datetime import datetime
    # save performance to output file
    date = datetime.now().date()
    i = 1
    while os.path.exists(TABLES_PATH + f'humor_results_params_{date}_{i}*.xlsx'):
        i += 1

    result_param_df.to_excel(TABLES_PATH + f'humor_results_params_{date}_{i}.xlsx')
    top_params_df.to_csv(TABLES_PATH + f'top_params_{date}_{i}.csv')


def fill_best_acc_df(result_param_df, df_averages):
    dataset_accs = {}
    top_params = []
    for dataset_name in TRAIN_DATASETS:
        df_curr_trained = df_averages[(df_averages['train_dataset'] == dataset_name)]

        # params_vals = {param_name: list(df_curr_trained[param_name].unique()) for param_name in params_column_names}

        # df_curr_trained.set_index(PARAMS_COLUMN_NAMES, inplace=True)
        # Return two parameters: dictionary of all params & accuracies, and dictionary of the best accuracies
        accs_and_best_accs = find_best_accuracies(df_curr_trained, dataset_name, which_best='Median')
        dataset_accs[dataset_name] = accs_and_best_accs
        # Iterate over the best accuracies and save all of them
        # calculate metrics' mean and std for all pairs of datasets
        all_accs = accs_and_best_accs[0]
        best_accs = accs_and_best_accs[1]
        for param_mertic, acc_and_params in best_accs.items():
            params = acc_and_params[1]
            df_params = df_curr_trained.at[params]
            metrics_dict = {metric_name: {} for metric_name in METRICS}

            for eval_dataset in DATASETS:
                df = df_params[(df_params['evaluate_dataset'] == eval_dataset)]
                # print(train_dataset, eval_dataset)
                for metric in METRICS:
                    # print(metric)
                    values = df[metric]
                    mean, std = values.mean(), values.std()
                    metrics_dict[metric][eval_dataset] = float("%.4f" % mean)
                    # print(mean, std)

            for metric in METRICS:
                result_param_df.loc[(metric, BASE_MODEL, dataset_name, param_mertic)] = metrics_dict[metric]

            top_3_accs = get_top_accuracies(all_accs, 3, 'median')
            print(f'Best 3 accuracies for dataset: {dataset_name}')
            for i, acc in enumerate(top_3_accs, 1):
                print(f'Top {i}')
                zipped_params = dict(zip(df_curr_trained.index.names, acc[0]))
                print(f'Params: {zipped_params}')
                print(f'Accuracy: {acc[1]}')

                zipped_params.update({'dataset': dataset_name, 'param_metric': param_mertic,
                                      'top': i})
                top_params.append(zipped_params)

    top_params_df = pd.DataFrame(top_params)
    top_params_df.set_index(['dataset', 'param_metric', 'top'], inplace=True)
    return top_params_df


if __name__ == '__main__':
    scores_df = load_score_table_from_files()

    df_averages = get_split_avg_df_all(scores_df)

    result_param_df = init_result_param_df()

    top_params_df = fill_best_acc_df(result_param_df, df_averages)

    save_result_param_df(result_param_df, top_params_df)

    pass
