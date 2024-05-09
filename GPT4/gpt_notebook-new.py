from openai import OpenAI
import tiktoken
import pandas as pd
import numpy as np


# creating gpt client
# TODO: login to openai client here
models = {'gpt-3.5-turbo': 'gpt-3.5-turbo',
          'gpt-4-turbo-1': 'gpt-4-turbo-preview',
          'gpt-4-turbo-2': 'gpt-4-1106-preview'}

model_name = models['gpt-4-turbo-2']

MAX_INPUT_TOKENS = 3800
NUM_OF_EXAMPLES = 5

# function for calculating num of tokens of sentence
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# load Reddit Dad Jokes dataset
df_dadjokes = pd.read_csv('../Data/new_humor_datasets/reddit_dadjokes/reddit_dadjokes.csv')
# get only 1 score samples
df_dadjokes = df_dadjokes[df_dadjokes.score == 1]


# clean from duplicates and reposts
df_dadjokes = df_dadjokes[df_dadjokes['joke'].apply(lambda joke: 'reposted' not in joke.lower())]
df_dadjokes.drop_duplicates(subset='joke', inplace=True)


lambda_num_tokens = lambda s: num_tokens_from_string(s, model_name)
df_dadjokes['num_tokens'] = df_dadjokes['joke'].apply(lambda_num_tokens)


# print(f'average num of tokens per sample = {np.mean(df_dadjokes.num_tokens)}')
# print(f'overall samples tokens per 10K samples = {10000 * np.mean(df_dadjokes.num_tokens)}')

def get_next_input(df, index):
    samples_count = 150
    df_for_input = df.iloc[index:index + samples_count]
    input_tokens_count = np.sum(df_for_input['num_tokens'])

    while input_tokens_count > MAX_INPUT_TOKENS:
        samples_count -= 10
        df_for_input = df_for_input.iloc[:samples_count]
        input_tokens_count = np.sum(df_for_input['num_tokens'])

    index += samples_count
    return df_for_input, index

# create examples and instruction to gpt request
examples_instructions_prompt = \
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
    " them non humorous. You can change anything but please change the least you can:"


index = 0
output_df = pd.DataFrame(columns=df_dadjokes.columns)
output_df['gpt4_output'] = None



# while index < 11000: # until we get to the final count
#     df_for_input, index  = get_next_input(df_dadjokes, index)
#     input_str = '\\n'.join([f'{i}. {joke}' for i, joke in enumerate(df_for_input, NUM_OF_EXAMPLES+1)])
#
#     # send to completion
#     completion = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {'role': 'user',
#              'content': examples_instructions_prompt + input_str}
#         ]
#     )

# print(examples_instructions_prompt + input_str)
# print(num_tokens_from_string(examples_instructions_prompt + input_str, model_name))

def create_df_from_completion(message):


print(completion.choices[0].message.content)


# In[ ]:


print(f'input tokens = {completion.usage.prompt_tokens}')
print(f'output tokens = {completion.usage.completion_tokens}')

