import pandas as pd
import os
import json

GENERIC_SYSTEM_PROMPT = "You are a helpful assistant"

PROMPT_TEMPLATE = "Below is an instruction that describes a task, paired with a text sample. Write a response that " \
                  "appropriately completes the request.\n" \
                  "<instruction>\n" \
                  "You are tasked with determining whether a given text is funny or not funny." \
                  " The text could be a news headline, an Amazon product description and related question," \
                  " a Yelp review, a one-liner joke, or any other short piece of text." \
                  " Your goal is to analyze the text and decide if it's intended to be humorous.\n" \
                  "</instruction>\n\n" \
                  "<text>\n" \
                  "{TEXT}\n" \
                  "</text>\n\n" \
                  "To determine if the text is funny, consider the following guidelines:\n" \
                  "1. Look for unexpected twists or surprises in the text\n" \
                  "2. Check for wordplay, puns, or clever use of language\n" \
                  "3. Identify any absurd or exaggerated elements\n" \
                  "4. Consider if the text subverts expectations or common assumptions\n" \
                  "5. Assess whether the text aims to amuse or entertain rather than inform\n\n" \
                  "Analyze the text carefully, keeping these guidelines in mind. Then, make a final classification" \
                  " on whether the text is funny or not funny. You don't need to provide your reasoning for why you" \
                  " believe the text is either funny or not funny. " \
                  "Write your classification as either 'Funny' or 'Not Funny' inside <classification> tags.\n\n\n" \
                  "Remember, humor can be subjective, so focus on whether the text appears to be intentionally" \
                  " crafted to be amusing or entertaining, rather than on whether you personally find it funny.\n" \
                  "Use HTML tags in your response.\nDo not add any words after </classification>."

OUTPUT_TEMPLATE = "<classification>\n{OUTPUT}\n</classification>"

DATA_PATH = '../Data/new_humor_datasets/balanced/{DATASET_NAME}/Splits/{SPLIT}.csv'
JSON_DATA_PATH = '../Data/new_humor_datasets/balanced/{DATASET_NAME}/Json/{SPLIT}.jsonl'
DATASETS = ['amazon', 'dadjokes', 'headlines', 'one_liners', 'yelp_reviews']


def generate_msg_list_wrapper(split):
    def generate_msg_list(row):
        input_prompt = PROMPT_TEMPLATE.format(TEXT=row['text'])
        output_prompt = OUTPUT_TEMPLATE.format(OUTPUT='Funny' if row['label'] == 1 else 'Not Funny')

        res = [{"role": "system", "content": GENERIC_SYSTEM_PROMPT},
               {"role": "user", "content": input_prompt}]

        if split != 'test':
            res.append({"role": "assistant", "content": output_prompt})

        return res

    return generate_msg_list


if __name__ == '__main__':
    for dataset in DATASETS:
        for split in ['train', 'val', 'test']:
            full_path = DATA_PATH.format(DATASET_NAME=dataset, SPLIT=split)
            df = pd.read_csv(full_path)
            msgs_series = df.apply(generate_msg_list_wrapper(split),
                                   axis=1)
            msgs_series = msgs_series.apply(lambda x: {"messages": x})

            json_filepath = JSON_DATA_PATH.format(DATASET_NAME=dataset, SPLIT=split)
            os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
            with open(json_filepath, 'w') as f:
                for d in msgs_series:
                    json.dump(d, f)
                    f.write('\n')
