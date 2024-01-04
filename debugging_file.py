import pandas as pd
import numpy as np

DATASETS = ['amazon', 'headlines', 'igg', 'twss']

if __name__ == '__main__':
    # processed_headlines_path = './Data/humor_datasets/headlines/with_val_fixed_train/{split}.csv'
    # original_headlines_path = './Data/original_datasets/headlines/{split}.csv'
    # processed_train_df = pd.read_csv(processed_headlines_path.format(split='train'))
    # processed_test_df = pd.read_csv(processed_headlines_path.format(split='test'))
    # original_train_df = pd.read_csv(original_headlines_path.format(split='train'))
    # original_test_df = pd.read_csv(original_headlines_path.format(split='test'))
    # # %%
    # original_all = original_train_df.append(original_test_df, ignore_index=True)
    #
    #
    # # %%
    # def add_mean_grade(row):
    #     origin_row = original_all[original_all['id'] == row['id']].squeeze()
    #     return origin_row['meanGrade']
    #
    #
    # # %%
    # processed_train_df['meanGrade'] = processed_train_df['label']
    # processed_train_df['meanGrade'] = processed_train_df.apply(add_mean_grade, axis=1)
    # processed_test_df['meanGrade'] = processed_test_df.apply(add_mean_grade, axis=1)

    # df = pd.read_csv('./files_from_cluster/pair_models.csv')

    #
    # single_datasets = ['amazon', 'headlines', 'igg', 'twss']
    # paired_datasets = ['amazon_headlines', 'amazon_igg', 'amazon_twss', 'headlines_igg', 'headlines_twss', 'igg_twss']
    # cols2 = ['performance', 'model', 'trained_on', 'seed', 'mean_accuracy', 'amazon', 'headlines', 'igg', 'twss']
    #
    # dataset_types = [('single', single_datasets), ('pair', paired_datasets)]
    # dataset_types = [('pair', paired_datasets)]
    # # dataset_types = [('single', single_datasets)]
    # path = './files_from_drive/'
    # model_name = 'bert_pair_09-19'
    #
    # for dataset_type in dataset_types:
    #     for ep in [3, 4]:
    #         for bs in [4, 8, 12]:
    #             # df = pd.read_csv(f'./files_from_cluster/t5_{dataset_type[0]}_09-19.csv')
    #             filename = f'{model_name} - ep={ep}_bs={bs}'
    #             df = pd.read_csv(path + filename + '.csv')
    #             output_df2 = pd.DataFrame(columns=cols2)
    #
    #             for dataset_train in dataset_type[1]:
    #                 df_train_dataset = df[df['trained_on'] == dataset_train]
    #                 if dataset_type[0] == 'single':
    #                     data_on_train_set = df_train_dataset[df_train_dataset['predict_on'] == dataset_train]
    #                 elif dataset_type[0] == 'pair':
    #                     compute_acc_on = dataset_train[:dataset_train.index('_')]
    #                     if compute_acc_on == 'headlines':
    #                         compute_acc_on = dataset_train[dataset_train.index('_') + 1:]
    #                     data_on_train_set = df_train_dataset[
    #                         df_train_dataset['predict_on'] == compute_acc_on]
    #                 best_accuracy_model = data_on_train_set[
    #                     data_on_train_set['accuracy'] == data_on_train_set['accuracy'].max()].squeeze()
    #                 if type(best_accuracy_model) == pd.core.frame.DataFrame:
    #                     best_accuracy_model = best_accuracy_model.iloc[0]
    #                 seed = best_accuracy_model['seed']
    #
    #                 df_with_seed = df_train_dataset[df_train_dataset['seed'] == seed]
    #                 accs_for_mean = data_on_train_set['accuracy']
    #
    #                 for performance in ['accuracy', 'recall', 'precision']:
    #                     row_to_df2 = {'performance': performance, 'model': df_with_seed['model'].iloc[0],
    #                                   'trained_on': df_with_seed['trained_on'].iloc[0],
    #                                   'seed': df_with_seed['seed'].iloc[0],
    #                                   'mean_accuracy': f'{np.mean(accs_for_mean)} +- {np.std(accs_for_mean)}'}
    #                     values = {dataset: df_with_seed[df_with_seed['predict_on'] == dataset][performance].squeeze()
    #                               for dataset in single_datasets}
    #                     row_to_df2.update(values)
    #                     output_df2 = output_df2.append([row_to_df2])
    #
    #             # output_df2.to_csv(f'./files_from_cluster/t5_{dataset_type[0]}_models_summary_09-19.csv', index=False)
    #             output_df2.to_csv(path + filename + '_summary.csv', index=False)

    output_path = './Data/output/results/'
    results_filename = 'results-2024-01-03.csv'
    results_fullpath = output_path + results_filename

    results_df = pd.read_csv(results_fullpath)
    seeds = results_df['seed'].unique()
    models = results_df['model'].unique()
    trained_datasets = results_df['trained_on'].unique()
    results_df_indexed = results_df.set_index(['model', 'trained_on', 'seed', 'predict_on'])

    cols = ['performance', 'model', 'trained_on', 'seed', 'accuracy_mean_std',
            'amazon', 'headlines', 'igg', 'twss']
    output_df = pd.DataFrame(columns=cols)

    for model in models:
        for trained_dataset in trained_datasets:
            curr_df = results_df[(results_df['model'] == model) &
                (results_df['trained_on'] == trained_dataset)]

            # check if the dataset is single/paired
            if '_' in trained_dataset:
                ds_to_accuracy = trained_dataset[:trained_dataset.index('_')]
                if ds_to_accuracy == 'headlines':
                    ds_to_accuracy = trained_dataset[trained_dataset.index('_') + 1:]
            else:
                ds_to_accuracy = trained_dataset

            df_compute_acc = curr_df[curr_df['predict_on'] == ds_to_accuracy]
            mean_acc = df_compute_acc['accuracy'].mean()
            std_acc = df_compute_acc['accuracy'].std()
            mean_std_acc = str("%.4f" % mean_acc) + '+-' + str("%.4f" % std_acc)

            best_acc_model = df_compute_acc[df_compute_acc['accuracy'] == df_compute_acc['accuracy'].max()].squeeze()
            if type(best_acc_model) == pd.core.frame.DataFrame:
                best_acc_model = best_acc_model.iloc[0]
            best_seed = best_acc_model['seed']

            df_with_seed = curr_df[curr_df['seed'] == best_seed]


            for performance in ['accuracy', 'recall', 'precision']:
                row_to_df = {'performance': performance, 'model': model,
                            'trained_on': trained_dataset, 'seed': best_seed,
                             'accuracy_mean_std': mean_std_acc}
                values = {dataset: df_with_seed[df_with_seed['predict_on'] == dataset][performance].iloc[0]
                          for dataset in DATASETS}
                row_to_df.update(values)
                # print(row_to_df2)
                output_df = output_df.append([row_to_df])

    output_df.to_csv(output_path + 'final-' + results_filename, index=False)


    pass