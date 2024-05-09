
from openai import OpenAI
import tiktoken
import pandas as pd
import numpy as np
from os.path import exists


# GLOBALS
MAX_INPUT_TOKENS = 3800
NUM_OF_EXAMPLES = 5

# create examples and instruction to gpt request
EXAMPLES_INSTRUCTIONS_PROMPT = \
    "###" \
    "1. input: 'A grizzly kept talking to me and annoyed me He was unbearable'" \
    "output: 'A grizzly kept talking to me and annoyed me He was intolerable'" \
    "2. input: 'For Christmas, I requested my family not to give me duplicates of the same item. Now I anticipate " \
    "receiving the missing sock next time.' " \
    "output: 'For Christmas, I requested my family not to give me duplicates of the same item. Now I anticipate " \
    "receiving the other book next time.' " \
    "3. input: 'My son’s fourth birthday was today, but when he came to see me I didn’t recognize him at first. I’d " \
    "never seen him be 4.' " \
    "output: My son’s fourth birthday was today, but when he came to see me I didn’t recognize him at first. He grew " \
    "up so fast. " \
    "4. input: 'I asked my friend if he liked Nickleback. He told me that he never gave me any money'" \
    "output: 'I asked my friend if he liked Nickleback. He told me that he prefers Kings of Leon.'" \
    "5. input: 'I went to a bookstore and asked where the self-help section was The clerk said that if she told me, " \
    "it would defeat the purpose.' " \
    "output: 'I went to a bookstore and asked where the self-help section was The clerk said it was in the third " \
    "aisle .' " \
    "###" \
    "Using the examples in ### markers, please change some of the words in the following sentences to make" \
    "them non humorous. Please keep the structure and the writing style " \
    " of the original sentence. You can change anything, but change the least you can:\n "


# function for calculating num of tokens of sentence
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_next_input(df, index):
    samples_count = 30
    df_for_input = df.iloc[index:index + samples_count]
    input_tokens_count = np.sum(df_for_input['num_tokens'])

    while input_tokens_count > MAX_INPUT_TOKENS:
        samples_count -= 10
        df_for_input = df_for_input.iloc[:samples_count]
        input_tokens_count = np.sum(df_for_input['num_tokens'])

    index += samples_count
    return df_for_input, index

# print(f'average num of tokens per sample = {np.mean(df_dadjokes.num_tokens)}')
# print(f'overall samples tokens per 10K samples = {10000 * np.mean(df_dadjokes.num_tokens)}')

# create series from completion output
def parse_response(response):
# response = completion.choices[0].message.content
    start_joke_idx = NUM_OF_EXAMPLES + 1
    end_joke_idx = NUM_OF_EXAMPLES + len(df_for_input)
    edited_jokes = []
    for i in range(start_joke_idx, end_joke_idx + 1):
        if i == end_joke_idx: # this is the last joke special case
            edited_joke = response[:]
        else:
            index_next_joke = response.index(f'{i+1}.')
            edited_joke = response[:index_next_joke]
            response = response[index_next_joke:]

        edited_joke = edited_joke[edited_joke.index(f'{i}. ') + len(str(i)) + 2:].strip() # remove 'i. ' from the joke
        if 'Output:' in edited_joke:
            edited_joke = edited_joke[edited_joke.index('Output:') + len('Output:'):].strip()
        edited_jokes.append(edited_joke)

    # for i, joke in enumerate(edited_jokes):
        # print(i, joke)
    edited_jokes_series = pd.Series(edited_jokes, name='edited_joke')

    return edited_jokes_series


# add series as column to df_for_input
def save_samples(df_for_input, edited_jokes_series, df_not_jokes):
    df_to_output = pd.concat([df_for_input.reset_index(drop=True), edited_jokes_series], axis=1, ignore_index=True)
    df_to_output.columns = df_not_jokes.columns

    # append to df_not_jokes and save
    df_not_jokes = pd.concat([df_not_jokes, df_to_output], axis=0, ignore_index=True)
    df_not_jokes.to_csv(path + not_jokes_filename, index=False)
    # df_not_jokes.to_csv(path + 'reddit_dadjokes_not_jokes_temp.csv', index=False)

    return df_not_jokes


if __name__ == '__main__':
    # creating gpt client
    # client = OpenAI()
    models = {'gpt-3.5-turbo': 'gpt-3.5-turbo',
              'gpt-4-turbo-1': 'gpt-4-turbo-preview',
              'gpt-4-turbo-2': 'gpt-4-1106-preview'}

    model_name = models['gpt-4-turbo-2']

    path = '../Data/new_humor_datasets/reddit_dadjokes/'
    # load Reddit Dad Jokes dataset (filtered and with id)
    df_dadjokes = pd.read_csv(path + 'reddit_dadjokes_with_id.csv')

    # get only 1 score samples
    df_dadjokes = df_dadjokes[df_dadjokes.score == 1]
    lambda_num_tokens = lambda s: num_tokens_from_string(s, model_name)
    df_dadjokes['num_tokens'] = df_dadjokes['joke'].apply(lambda_num_tokens)
    # filter very long jokes
    df_dadjokes = df_dadjokes[df_dadjokes['num_tokens'] < 50]

    # check if file exists
    not_jokes_filename = 'reddit_dadjokes_not_jokes.csv'
    if exists(path + not_jokes_filename):
        df_not_jokes = pd.read_csv(path + not_jokes_filename)
        df_not_jokes.dropna(inplace=True)
    else:
        df_not_jokes = pd.DataFrame(columns=df_dadjokes.columns)
        df_not_jokes['edited_joke'] = None

    curr_index = len(df_not_jokes)
    curr_batch = 0

    while curr_index < 11000: # until we get to the final count
        # curr_index = 99
        # df_for_input = df_dadjokes.iloc[curr_index:curr_index + 30]
        df_for_input, curr_index = get_next_input(df_dadjokes, curr_index)
        input_str = '\n'.join([f'{i}. {joke.strip()}' for i, joke in enumerate(df_for_input['joke'], NUM_OF_EXAMPLES+1)])
        # print(input_str)

        # send to completion
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user',
                 'content': EXAMPLES_INSTRUCTIONS_PROMPT + input_str}
            ]
        )

        edited_jokes_series = parse_response(completion.choices[0].message.content)
        df_not_jokes = save_samples(df_for_input, edited_jokes_series, df_not_jokes)
        print(f'{curr_batch}. current amount of non-jokes = {curr_index}')
        curr_batch += 1

# print(completion.choices[0].message.content)
# print(EXAMPLES_INSTRUCTIONS_PROMPT + input_str)
# print(num_tokens_from_string(EXAMPLES_INSTRUCTIONS_PROMPT + input_str, model_name))

# print(f'input tokens = {completion.usage.prompt_tokens}')
# print(f'output tokens = {completion.usage.completion_tokens}')

