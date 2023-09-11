import pandas as pd



if __name__ == '__main__':
    processed_headlines_path = './Data/humor_datasets/headlines/with_val_fixed_train/{split}.csv'
    original_headlines_path = './Data/original_datasets/headlines/{split}.csv'
    processed_train_df = pd.read_csv(processed_headlines_path.format(split='train'))
    processed_test_df = pd.read_csv(processed_headlines_path.format(split='test'))
    original_train_df = pd.read_csv(original_headlines_path.format(split='train'))
    original_test_df = pd.read_csv(original_headlines_path.format(split='test'))
    # %%
    original_all = original_train_df.append(original_test_df, ignore_index=True)


    # %%
    def add_mean_grade(row):
        origin_row = original_all[original_all['id'] == row['id']].squeeze()
        return origin_row['meanGrade']


    # %%
    processed_train_df['meanGrade'] = processed_train_df['label']
    processed_train_df['meanGrade'] = processed_train_df.apply(add_mean_grade, axis=1)
    processed_test_df['meanGrade'] = processed_test_df.apply(add_mean_grade, axis=1)

