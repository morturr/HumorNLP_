from openai import OpenAI
import tiktoken
import pandas as pd
import numpy as np

models = {'gpt-3.5-turbo': 'gpt-3.5-turbo',
          'gpt-4-turbo-1': 'gpt-4-turbo-preview',
          'gpt-4-turbo-2': 'gpt-4-1106-preview'}

model_name = models['gpt-4-turbo-1']


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# calculate average tokens count for sample
# load Reddit Dad Jokes dataset
df_dadjokes = pd.read_csv('../Data/new_humor_datasets/reddit_dadjokes/reddit_dadjokes.csv')
# get only 1 score samples
df_dadjokes = df_dadjokes[df_dadjokes.score == 1]

lambda_num_tokens = lambda s: num_tokens_from_string(s, model_name)
df_dadjokes['num_tokens'] = df_dadjokes['joke'].apply(lambda_num_tokens)

print(f'average num of tokens per sample = {np.mean(df_dadjokes.num_tokens)}')
print(f'overall samples tokens per 10K samples = {10000 * np.mean(df_dadjokes.num_tokens)}')


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

input_prompt = \
    "6. 'People often ask me how is it like working as an elevator operator I say there’s nothing special. It " \
    "has its ups and downs' " \
    "7. 'How do you make the number one disappear? Just add a g and it's gone.'" \
    "8. 'Why do flamingos lift one leg? If they lifted both, they'd fall.'" \
    "9. 'I use the word Mucho around my Spanish friends It means a lot to them'" \
    "10. 'I told my kids my super-hero name would be: Mr. Pee Pee because, if you see me, ... urine trouble!'"

print(f'instruction prompt tokens = {num_tokens_from_string(examples_instructions_prompt, model_name)}')
print(f'input prompt tokens = {num_tokens_from_string(input_prompt, model_name)}')

# completion = client.chat.completions.create(
#     model=model_name,
#     messages=[
#         {'role': 'user',
#          'content': "###"
#                     "1. input: 'A grizzly kept talking to me and annoyed me He was unbearable'"
#                     "output: 'A grizzly kept talking to me and annoyed me He was intolerable'"
#                     "2. input: 'For Christmas, I requested my family not to give me duplicates of the same item. Now I anticipate "
#                     "receiving the missing sock next time.' "
#                     "output: 'For Christmas, I requested my family not to give me duplicates of the same item. Now I anticipate "
#                     "receiving the other book next time.' "
#                     "3. input: 'My son’s fourth birthday was today, but when he came to see me I didn’t recognize him at first. I’d "
#                     "never seen him be 4.' "
#                     "output: My son’s fourth birthday was today, but when he came to see me I didn’t recognize him at first. He grew "
#                     "up so fast. "
#                     "4. input: 'I asked my friend if he liked Nickleback. He told me that he never gave me any money'"
#                     "output: 'I asked my friend if he liked Nickleback. He told me that he prefers Kings of Leon.'"
#                     "5. input: 'I went to a bookstore and asked where the self-help section was The clerk said that if she told me, "
#                     "it would defeat the purpose.' "
#                     "output: 'I went to a bookstore and asked where the self-help section was The clerk said it was in the third "
#                     "aisle .' "
#                     "###"
#                     "Using the examples in ### markers, please change some of the words in the following sentences to make"
#                     " them non humorous. You can change anything but please change the least you can:"
#                     "6. input: 'People often ask me how is it like working as an elevator operator I say there’s nothing special. It "
#                     "has its ups and downs' "
#                     "7. input: 'How do you make the number one disappear? Just add a g and it's gone.'"
#                     "8. input: 'Why do flamingos lift one leg? If they lifted both, they'd fall.'"
#                     "9. input: 'I use the word Mucho around my Spanish friends It means a lot to them'"
#                     "10. input: 'I told my kids my super-hero name would be: Mr. Pee Pee because, if you see me, ... urine trouble!'"}
#     ])

# print(completion.choices[0].message.content)
# print(f'input tokens = {completion.usage.prompt_tokens}')
# print(f'output tokens = {completion.usage.completion_tokens}')

'''
original: ‘I surprised my friend with a new car. I also surprised my friend with cancer.

I just like surprising people.’

edited: ‘I surprised my friend with a new car. I also surprised my friend with a cake.

I like surprising people.’

original: ‘People often ask me how is it like working as an elevator operator I say there’s nothing special. It has its ups and downs.’

edited: ‘People often ask me how is it like working as an elevator operator I say there’s nothing special. It has its difficulties.’ 
'''
'''
    {"role": "system", "content": "You will take task instructions from the ###instruction:``### section and use"
                                  " those to process data in the ###input:``### section using the ###output:``### "
                                  "section as a template"},
    {"role": "user", "content": "###instruction: 'Change some of the words from the following sentences"
                                "to make them non humorous. You can change anything, but change"
                                "the least you can.'###"}
    {"role": "user", "content": "###output:'{user_output_template}'###"}
    {"role": "user", "content": "###input: '{user_query_input}'###"}
'''
