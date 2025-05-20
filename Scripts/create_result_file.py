import random

import pandas as pd
import os

from typing import Tuple, Dict, List, Any

from pandas import DataFrame

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_type", dest="experiment_type", type=str, required=True)
parser.add_argument("--use_model", dest="use_model", type=str, required=True)

args = parser.parse_args()

# USE_MODEL = 'Llama-2'
# USE_MODEL = 'Mistral'
USE_MODEL = args.use_model

PATH = f'../{USE_MODEL}/Results'
TABLES_PATH = PATH + '/Tables/'

FINAL_RESULTS_PATH = PATH + '/Final Results'
FINAL_RESULTS_TABLES_PATH = TABLES_PATH + 'Final Tables/'
PATH = FINAL_RESULTS_PATH
TABLES_PATH = FINAL_RESULTS_TABLES_PATH


# # path for hyperparameter results
# HYPERPARAMS_PATH = PATH + '/Hyperparameter search'

DATASETS = ['amazon', 'dadjokes', 'headlines', 'one_liners', 'yelp_reviews']
PAIR_DATASETS = ['amazon_dadjokes', 'amazon_headlines', 'dadjokes_headlines',
                 'dadjokes_one_liners', 'headlines_one_liners', 'headlines_yelp_reviews',
                 'one_liners_amazon', 'one_liners_yelp_reviews',
                 'yelp_reviews_amazon', 'yelp_reviews_dadjokes']
# DATASETS = ['amazon']

def load_experiment_prefix():
    """
    This function loads the experiment prefix from the command line arguments
    :return: EXP_PREFIX - the experiment prefix
    """
    exp_prefix = ''
    if args.experiment_type == 'DEFAULT':
        exp_prefix = ''
    elif args.experiment_type == 'LOO':
        exp_prefix = 'LOO_'
    elif args.experiment_type == 'LOO_WITH_FEW':
        exp_prefix = 'LOO_WITH_FEW_'
    elif args.experiment_type == 'HYPERPARAMS':
        exp_prefix = ''
    elif args.experiment_type == 'PAIR':
        exp_prefix = 'PAIR_'
    else:
        raise ValueError(f'Unknown experiment type: {args.experiment_type}')

    return exp_prefix

def load_path_template() -> str:
    """
    This function loads the path template from the command line arguments
    :return: path_template - the path template
    """
    global PATH

    if args.experiment_type == 'LOO':
        path_template = PATH + '/Leave-One-Out/{dataset}-models/'
    elif args.experiment_type == 'LOO_WITH_FEW':
        path_template = PATH + '/LOO-with-Few/{dataset}-models/'
    elif args.experiment_type == 'HYPERPARAMS':
        path_template = PATH + '/Hyperparameter search/{dataset}-models/'
    elif args.experiment_type == 'PAIR':
        path_template = PATH + '/Pairs/{dataset}-models/'
    elif args.experiment_type == 'DEFAULT':
        path_template = PATH + '/Singles/{dataset}-models/'
    else:
        raise ValueError(f'Unknown experiment type: {args.experiment_type}')

    return path_template

def load_train_datasets():
    '''
    This function loads the train datasets according to the experiment prefix
    :return:
    '''
    global TRAIN_DATASETS

    exp_prefix = load_experiment_prefix()
    if exp_prefix == 'PAIR_':
        TRAIN_DATASETS = [f'{exp_prefix}{dataset}' for dataset in PAIR_DATASETS]
    else:
        TRAIN_DATASETS = [f'{exp_prefix}{dataset}' for dataset in DATASETS]


def init_config():
    """
    This function initializes the config file with the relevant paths and parameters
    :return:
    """
    global PATH_TEMPLATE


    PATH_TEMPLATE = load_path_template()
    load_train_datasets()


init_config()



METRICS = ['accuracy', 'f1', 'recall', 'precision']

if USE_MODEL == 'Llama-2':
    BASE_MODEL = 'llama-2-7b'
elif USE_MODEL == 'Mistral':
    BASE_MODEL = 'Mistral-7B-v0.1'
else:
    raise ValueError(f'Unknown model: {USE_MODEL}')

AVERAGE_COLUMNS_BY = 'seed'
# AVERAGE_COLUMNS_BY = 'split_num'

if AVERAGE_COLUMNS_BY == 'split_num':
    PARAMS_COLUMN_NAMES = ['seed', 'learning_rate', 'per_device_train_batch_size',
                       'per_device_eval_batch_size', 'max_steps', 'lora_rank', 'lora_alpha', ]
elif AVERAGE_COLUMNS_BY == 'seed':
    # Because we want to average by seed, we need to remove the seed from the params columns
    # so it won't be included in the groupby index when computing the average
    PARAMS_COLUMN_NAMES = ['learning_rate', 'per_device_train_batch_size',
                       'per_device_eval_batch_size', 'max_steps', 'lora_rank', 'lora_alpha', ]



def load_score_table_from_files() -> pd.DataFrame:
    """
    This function loads all the results from the saved models files in PATH_TEMPLATE
    and returns a dataframe with all the results
    Used for the regular models and not for leave-one-out models
    :return: df_scores - a dataframe with all the results
    """
    df_scores: DataFrame = pd.DataFrame()

    for dataset_name in TRAIN_DATASETS:
        root_dir = PATH_TEMPLATE.format(dataset=dataset_name)
        for _, dirs, _ in os.walk(root_dir):
            for dir_name in dirs:
                inner_dir = os.path.join(root_dir, dir_name)
                score_file_path = os.path.join(inner_dir, f'{dataset_name}_scores.csv')
                df = pd.read_csv(score_file_path)
                # for the first df, set its columns to be the columns of the overall dataframe (scores_df)
                if df_scores.empty:
                    df_scores = df
                else:
                    df_scores = pd.concat([df_scores, df])

    return df_scores


def get_top_accuracies(accuracies, get_top=3, get_by='median'):
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1][get_by], reverse=True)
    return sorted_accuracies[:get_top]


def find_best_accuracies(df: pd.DataFrame, trained_dataset_name: str, which_best: str='All') \
        -> (Tuple[Dict[Tuple[int], Dict[str, int]]], Dict[str, Tuple[int, Tuple[int]]]):
    """
    This function receives a dataframe with the results of a single dataset,
     averaged over the cross validation splits.
     It calculates the (averaged?) accuracies of every set of parameters,
     and returns a dictionary of the parameters and their accuracies,
     and a dictionary of the best accuracies overall:
      metric name (mean/median/trained): (accuracy, parameters)

    :param df: a dataframe with the results of a single dataset
    :param trained_dataset_name: the name of the dataset that the results of df belong to (the trained dataset)
    :param which_best: which best accuracy to return: 'All' for all, 'Median' for the best median accuracy
    :return: accuracies: a dictionary of the parameters and their accuracies
    :return: best_accuracies: a dictionary of the best accuracies overall
    """

    accuracies = {}
    best_trained, best_median = (-1, None), (-1, None)

    trained_dataset_names = [dataset_name for dataset_name in DATASETS if dataset_name in trained_dataset_name]

    for params, rows in df.groupby(by=PARAMS_COLUMN_NAMES):
        if len(trained_dataset_names) == 1:
            row_trained = rows[rows['evaluate_dataset'] == trained_dataset_names[0]]
        elif len(trained_dataset_names) > 1:
            row_trained = rows[rows['evaluate_dataset'].isin(trained_dataset_names)]
        # row_others = rows[rows['evaluate_dataset'] != trained_dataset_name]
        curr_trained_accuracy = row_trained['accuracy'].mean()
        others_accuracy_median = rows['accuracy'].median()

        best_trained = (curr_trained_accuracy, params) if curr_trained_accuracy > best_trained[0] else best_trained
        best_median = (others_accuracy_median, params) if others_accuracy_median > best_median[0] else best_median

        curr_accs = {'trained': curr_trained_accuracy,
                     'median': others_accuracy_median}

        accuracies[params] = curr_accs

    # print(f'Accuracies for dataset: {trained_dataset_name}')
    # print(f'Best trained accuracy = {best_trained}')
    # print(f'Best median accuracy = {best_median}')

    if which_best == 'All':
        return accuracies, {'best_trained': best_trained,
                            'best_median': best_median}

    # Return only best median
    if which_best == 'Median':
        return accuracies, {'best_median': best_median}


def init_result_param_df():
    """
    This function initializes a dataframe with the template of the results
    :return: df: a dataframe with the template of the results
    """
    df = pd.read_excel(TABLES_PATH + 'result_param_metric_template.xlsx')
    df.fillna(method='ffill', axis=0, inplace=True)
    df.set_index(['metric', 'model', 'trained on', 'param_metric'], inplace=True)

    return df


def get_avg_single_df(df: pd.DataFrame, average_by: str) -> pd.DataFrame:
    """
    This function receives a dataframe with the results of a single dataset
     and returns a dataframe with the average over the cross validation splits or seeds
    :param df: a dataframe with the results of a single dataset
    :param average_by: the column to average by (seed or split_num)
    :return: df_avg - a dataframe of a single dataset with the average over the cross validation splits or seeds
    """

    # Iterate over the existing parameters combinations and create df of the average cv splits/seeds results
    df_avg = pd.DataFrame(columns=df.columns)

    # Columns you want to average the scores for
    columns_to_average = ['accuracy', 'precision', 'recall', 'f1']


    #TODO Mor For Final: comment the command below and comment out two rows below beacuse final files don't have params
    for params, rows in df.groupby(by=PARAMS_COLUMN_NAMES):
    # for rows in [df]: ## for final files who don't have params
        if len(rows) != 20:
            print(f'Error in dataset {rows.iloc[0]["train_dataset"]}:\n'
                  f' Expected 20 rows for params {params}, got {len(rows)}')
            rows_datasets_lst = rows['dataset'].unique()
            for dataset in rows_datasets_lst:
                assert len(rows[rows['dataset'] == dataset]) == 20,\
                    f'Expected 20 rows for params {params} (4 seeds, 5 evaluate datasets), got {len(rows[rows["dataset"] == dataset])}'
            selected_dataset = rows_datasets_lst[random.randint(0, len(rows_datasets_lst) - 1)]
            print(f'** There are several datasets = {rows_datasets_lst} with same params = {params}. Using results of {selected_dataset} **')
            rows = rows[rows['dataset'] == selected_dataset]

            # raise AssertionError(f'Expected 20 rows for params {params}, got {len(rows)}')
        # assert len(rows) == 20, f'Expected 20 rows for params {params} (4 seeds, 5 evaluate datasets), got {len(rows)}'
        # assert len(rows) == 5, f'Expected 5 rows for params {params} (1 seed 1 split, 5 evaluate datasets), got {len(rows)}'
        # if len(rows) != 5:
        #     continue

        for dataset_name in DATASETS:
            rows_of_dataset = rows[rows['evaluate_dataset'] == dataset_name]

            # Calculate the mean for the relevant columns
            mean_values = rows_of_dataset[columns_to_average].mean()

            # Prepare a new row with average values and other relevant info
            # For example, 'split' can be labeled as 'average'
            new_row = rows_of_dataset.iloc[0].copy(deep=True)

            new_row.update(mean_values)
            new_row.update({average_by: 'average'})
            new_row_df = pd.DataFrame([new_row], columns=df.columns)

            if df_avg.empty:
                df_avg = new_row_df
            else:
                df_avg = pd.concat([df_avg, new_row_df], axis=0)

    return df_avg


def get_avg_df_all(df: pd.DataFrame, average_by: str) -> pd.DataFrame:
    """
    This function receives a dataframe with all the results
     and returns a dataframe with the average over all the cross validation splits or seeds results
    :param df: a dataframe with all the results
    :param average_by: the column to average by (seed or split)
    :return: df_avg_all - a dataframe with the average over all the cross validation splits or seeds results
    """

    df_avg_all = pd.DataFrame()
    for dataset_name in TRAIN_DATASETS:
        # for each dataset separately,
        # create a dataframe with the average over all the cross validation splits results
        # set the parameters as the index as all the cv splits need to be averaged by the same parameters
        df_curr_trained = df[(df['train_dataset'] == dataset_name)]

        # TODO Mor For Final: comment command below because no params in final files
        df_curr_trained.set_index(PARAMS_COLUMN_NAMES, inplace=True)

        try:
            df_curr_trained_avg = get_avg_single_df(df_curr_trained, average_by)
        except AssertionError as e:
            print(f'Error in dataset {dataset_name}: {e}')
            continue

        df_curr_trained_avg.index.names = df_curr_trained.index.names

        if df_avg_all.empty:
            df_avg_all = df_curr_trained_avg

        else:
            df_avg_all = pd.concat([df_curr_trained_avg, df_avg_all])

    return df_avg_all


def save_result_param_df(result_param_df: pd.DataFrame, top_params_df: pd.DataFrame) -> None:
    """
    This function receives a dataframe with the best results of all datasets with the output file results template,
    and a dataframe of top parameters of each dataset,
    and saves them.

    :param result_param_df: a dataframe with the best results of all datasets with the output file results template
    :param top_params_df: a dataframe of top parameters of each dataset
    :return:
    """
    from datetime import datetime
    # save performance to output file
    date = datetime.now().date()
    i = 1
    while os.path.exists(TABLES_PATH + f'humor_results_params_{date}_{i}.xlsx'):
        i += 1

    result_param_df.to_excel(TABLES_PATH + f'humor_results_params_{date}_{i}.xlsx')
    top_params_df.to_csv(TABLES_PATH + f'top_params_{date}_{i}.csv')


def save_best_accuracies(result_param_df: pd.DataFrame, df_trained: pd.DataFrame,
                         dataset_name: str, best_accs: Dict[str, Tuple[int, Tuple[int]]]) -> None:
    """
    This function receives a dataframe with the results of a single dataset,
    averaged over the average_by (seeds or cv splits), and a dictionary of the best accuracies.
    It calculates the mean & std accuracies for each dataset and metric and saves them in the result_param_df.
    :param result_param_df: an empty dataframe with the template of the result output file
    :param df_trained: a dataframe with the results of a single dataset, averaged over the cv splits / seeds
    :param dataset_name: the name of the dataset that the results of df belong to (the trained dataset)
    :param best_accs: a dictionary of the best accuracies
    :return:
    """

    # Iterate over the best accuracies and save all of them
    # calculate metrics' mean and std for all pairs of datasets
    # TODO Mor For Final:  replace commented code with below (until metrics_dict=)
    #  because final results don't have params and best accs
    # for _ in range(1): # for final
    #     df_of_params = df_trained
    #     metrics_dict = {metric_name: {} for metric_name in METRICS}
    #     acc_selection_mertic = 'final results'
    for acc_selection_mertic, acc_and_params in best_accs.items():
        params = acc_and_params[1]

        df_of_params = df_trained.at[params]
        metrics_dict = {metric_name: {} for metric_name in METRICS}

        # for each accuracy metric, calculate the mean and std of METRICS over all datasets
        for eval_dataset in DATASETS:
            df = df_of_params[(df_of_params['evaluate_dataset'] == eval_dataset)]

            for metric in METRICS:
                values = df[metric]
                mean, std = values.mean(), values.std()
                metrics_dict[metric][eval_dataset] = float("%.4f" % mean)

        for metric in METRICS:
            result_param_df.loc[(metric, BASE_MODEL, dataset_name, acc_selection_mertic)] = metrics_dict[metric]

def get_top_params(df_trained: pd.DataFrame, dataset_name: str,
                   best_accs: Dict[str, Tuple[int, Tuple[int]]], all_accs: Tuple[Dict[Tuple[int], Dict[str, int]]])\
        -> List[Dict[str, Any]]:
    """
    This function receives a dataframe with the results of a single dataset,
    averaged over the cross validation splits, and a dictionary of the best accuracies by the accuracies metrics,
    and a dictionary of all the accuracies by the parameters.
    It calculates the top 3 accuracies for each dataset and return them in top_params.
    :param df_trained: a dataframe with the results of a single dataset, averaged over the cross validation splits
    :param dataset_name: the name of the dataset that the results of df belong to (the trained dataset)
    :param best_accs: a dictionary of the best accuracies by the accuracies metrics
    :param all_accs: a dictionary of all the accuracies by the parameters
    :return: top_params: a list of the top 3 accuracies for each dataset
    """

    top_params = []
    for acc_selection_mertic, acc_and_params in best_accs.items():

        # get top 3 accuracies for each dataset and save them in top_params
        get_top_by = acc_selection_mertic[len('best_'):]
        # top_3_accs = get_top_accuracies(all_accs, 3, 'median')
        top_3_accs = get_top_accuracies(all_accs, 3, get_top_by)

        print(f'Best 3 accuracies for dataset: {dataset_name}')
        for i, acc in enumerate(top_3_accs, 1):
            print(f'Top {i}')
            params = acc[0]

            df_of_params = df_trained.at[params]
            # Check that this set of parameters is unique to one model
            if df_of_params['model_name'].nunique() == 1:
                model_name = df_of_params['model_name'].values[0]
            else:
                raise Exception(f'Error: multiple model names for the same set of parameters:'
                                f' {params}, models: {df_of_params["model_name"].unique()}')

            zipped_params = dict(zip(df_trained.index.names, acc[0]))
            print(f'Params: {zipped_params}')
            print(f'Accuracy: {acc[1]}')

            zipped_params.update({'dataset': dataset_name, 'param_metric': acc_selection_mertic,
                                  'top': i, 'median_accuracy': "%.4f" %  acc[1][get_top_by],
                                  'model_name': model_name})
            top_params.append(zipped_params)

    return top_params

def create_accuracy_top_params_all_datasets(result_param_df: pd.DataFrame, df_averages: pd.DataFrame) -> pd.DataFrame:
    """
    This function receives a dataframe with the results of all datasets, averaged over all the cross validation splits,
    and an empty dataframe with the template of the result output file.
    It calculates the best accuracies for each dataset and saves them in the result_param_df.
    It also returns a dataframe with the top 3 accuracies parameters for each dataset.

    :param result_param_df: an empty dataframe with the template of the result output file
    :param df_averages: a dataframe with the results of all datasets, averaged over all the cross validation splits
    :return: top_params_df: a dataframe with the top 3 accuracies parameters for each dataset
    """
    all_top_params = []
    for dataset_name in TRAIN_DATASETS:
        df_curr_trained = df_averages[(df_averages['train_dataset'] == dataset_name)]

        # Return two parameters: dictionary of all params & accuracies, and dictionary of the best accuracies
        # TODO Mor For Final: comment the commands below because final results don't have params and best accs
        # # TODO Mor: change to get all best, and take the best by trained accuracy
        accs_and_best_accs = find_best_accuracies(df_curr_trained, dataset_name, which_best='Median')
        # # accs_and_best_accs = find_best_accuracies(df_curr_trained, dataset_name, which_best='All')

        all_accs = accs_and_best_accs[0]
        best_accs = accs_and_best_accs[1]

        # TODO Mor For Final: replace the command below because final results don't have params and best accs
        # save_best_accuracies(result_param_df, df_curr_trained, dataset_name, None) # for final
        save_best_accuracies(result_param_df, df_curr_trained, dataset_name, best_accs)
        curr_dataset_top_params = get_top_params(df_curr_trained, dataset_name, best_accs, all_accs)
        all_top_params.extend(curr_dataset_top_params)

    # TODO Mor For Final: comment the code below because final results don't have params
    # return pd.DataFrame() # for final
    top_params_df = pd.DataFrame(all_top_params)
    top_params_df.set_index(['dataset', 'param_metric', 'top'], inplace=True)
    return top_params_df


if __name__ == '__main__':
    scores_df = load_score_table_from_files()

    df_averages = get_avg_df_all(scores_df, average_by=AVERAGE_COLUMNS_BY)

    result_param_df = init_result_param_df()

    top_params_df = create_accuracy_top_params_all_datasets(result_param_df, df_averages)

    save_result_param_df(result_param_df, top_params_df)

    pass
