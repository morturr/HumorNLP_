import os

from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import argparse
import json
import pandas as pd

import sys

sys.path.append('../../')
sys.path.append('../../../')
from FlanT5.classify_and_evaluate import create_report

import time

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", dest="train_dataset", type=str, required=True)
parser.add_argument("--eval_datasets", dest="eval_datasets", type=str,
                    nargs='+', default=['amazon', 'dadjokes', 'headlines',
                                        'one_liners', 'yelp_reviews'])
parser.add_argument("--inference_batch_size", dest="inference_batch_size", type=int, default=4)
parser.add_argument("--data_percent", dest="data_percent", type=float, default=100)
parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=20)
parser.add_argument("--model_name", dest="model_name", type=str, default='')
parser.add_argument("--lora_path", dest="lora_path", type=str, default='')
parser.add_argument("--results_path", dest="results_path", type=str, default='')
parser.add_argument("--tokenizer_path", dest="tokenizer_path", type=str, default='')
args = parser.parse_args()


def predict_wrapper(model, tokenizer):
    def predict(input):
        completion_request = ChatCompletionRequest(messages=input)

        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=args.max_new_tokens, temperature=0.0,
                                 eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        if len(out_tokens) > 0:
            out_tokens = out_tokens[0]

        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens)
        return result

    return predict


def verify_text_matches(test_df, json_df):
    def extract_text_from_user_message(row):
        content = row[1]['content']
        prefix = '<text>\n'
        suffix = '\n</text>'
        text = content[content.find(prefix) + len(prefix):content.rfind(suffix)]
        return text

    json_df['content'] = json_df['messages'].apply(extract_text_from_user_message)
    test_df['is_contained'] = test_df.apply(lambda row: row['text'] in json_df.loc[row.name, 'content'], axis=1)
    return test_df['is_contained'].all()


def get_test_labels(eval_dataset_name, json_df):
    data_path = '/cs/labs/dshahaf/mortur/HumorNLP_/Data/new_humor_datasets/balanced/{DATASET}/Splits/test.csv'

    data_path = data_path.format(DATASET=eval_dataset_name)
    df = pd.read_csv(data_path)
    df = df[:len(json_df)]

    assert verify_text_matches(df, json_df), f'{eval_dataset_name}: Texts do not match'
    return df['label'].tolist()


def prediction_to_label(prediction):
    prefix = '<classification>\n'
    suffix = '\n</classification>'
    prediction = prediction[prediction.find(prefix) + len(prefix):prediction.rfind(suffix)].strip()

    label = 1 if prediction == 'Funny' else \
            0 if prediction == 'Not Funny' else \
            -1

    return label


if __name__ == '__main__':
    tokenizer = MistralTokenizer.from_file(args.tokenizer_path)  # change to extracted tokenizer file
    model = Transformer.from_folder(
        "/cs/labs/dshahaf/mortur/HumorNLP_/Mistral/mistral_models/7B")  # change to extracted model dir
    model.load_lora(args.lora_path)
    assert args.train_dataset in args.model_name, 'Model and train dataset do not match'

    predict = predict_wrapper(model, tokenizer)

    for eval_dataset_name in args.eval_datasets:
        start_time = time.time()
        print('Evaluating on: ', eval_dataset_name)
        file_path = '/cs/labs/dshahaf/mortur/HumorNLP_/Data/new_humor_datasets/balanced/{dataset}' \
                    '/Json/test.jsonl'.format(dataset=eval_dataset_name)

        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # data = data[:10]
        df = pd.DataFrame.from_records(data, columns=['messages'])

        # Make predictions
        df['prediction'] = df['messages'].apply(predict)

        # Get true labels and predicted labels
        df['true_label'] = get_test_labels(eval_dataset_name, df)
        df['predicted_label'] = df['prediction'].apply(prediction_to_label)

        if os.path.exists('../../Results'):
            output_dir = f'../../Results/{args.model_name}'
        elif os.path.exists('../Results'):
            output_dir = f'../Results/{args.model_name}'
        else:
            output_dir = f'{args.model_name}'
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(f'{output_dir}/{eval_dataset_name}_predictions.csv', index=False)

        run_args = {'train_dataset': args.train_dataset,
                    'evaluate_dataset': eval_dataset_name,
                    'model_name': args.model_name,
                    'save_dir': output_dir}

        prediction_list_int = df['predicted_label'].to_list()
        label_list_int = df['true_label'].to_list()

        # Remove illegal predictions
        illegal_indices = [i for i, val in enumerate(prediction_list_int) if val == -1]
        if illegal_indices:
            print(f'Illegal predictions found in {len(illegal_indices)} examples')
            for index in illegal_indices[::-1]:
                prediction_list_int.pop(index)
                label_list_int.pop(index)

        create_report(label_list_int, prediction_list_int, run_args, pos_label=1)

        end_time = time.time()
        time_length = end_time - start_time
        print('Evaluation time: ', time_length / 60, ' minutes')

# prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."
#
# completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
#
# tokens = tokenizer.encode_chat_completion(completion_request).tokens
#
# out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
# result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
# print(result)
#
# out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
# result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
#
# print(result)
