import google.generativeai as genai
from FlanT5.data_loader import load_dataset
from sklearn.metrics import (
    classification_report,)

from huggingface_hub import InferenceClient


def get_response(row):
    # output = 'Yes' if row['label'] == 1 else 'No'
    # row['instruction'] = row['instruction'] + output
    # instruction = row['instruction']
    # terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
    # # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
    row['text label'] = 'Yes' if row['label'] == 1 else 'No'
    # row['instruction'] = instruction[instruction.index(terminator) + len(terminator):]
    return row


def extract_response(response):
    if '### Response:\n' not in response:
        return 'Illegal'
    response_idx = response.index('### Response:\n') + len('### Response:\n')
    extracted_response = response[response_idx:]
    return extracted_response


def get_prediction(extracted_response):
    if 'not funny' in extracted_response.lower():
        return 'No'
    if 'funny' in extracted_response.lower() and 'not' not in extracted_response.lower():
        return 'Yes'

    return 'Illegal'

# genai.configure() # add key if needed

# model = genai.GenerativeModel("gemini-2.5-flash-lite")

client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct", token='')

for eval_dataset_name in ['amazon', 'dadjokes', 'headlines', 'one_liners']:
    curr_dataset = load_dataset(eval_dataset_name, instruction_version=0,
                                                test_percent=0.033, add_instruction=True, with_val=False,
                                                percent=0.1)


    print(f'@@@ Evaluating on {eval_dataset_name} dataset. Examples count = {len(curr_dataset["test"])} @@@')


    true_labels = curr_dataset['test'].map(get_response)

    # Remove the response from the instruction (If there is one)
    # curr_dataset[split] = curr_dataset[split].map(remove_response)

    true_labels = [true_labels[i]['text label'] for i in range(len(true_labels))]
    texts = [curr_dataset['test'][i]['instruction'] for i in range(len(curr_dataset['test']))]

    all_responses = []
    prediction_list, label_list = [], []

    for i in range(0, len(texts)):
        batch_texts = texts[i]
        batch_labels = true_labels[i]
        # output = model.generate_content(batch_texts)
        # response = output.text.strip()
        response = client.text_generation(batch_texts, max_new_tokens=10, do_sample=False)

        all_responses.append(response)
        batch_predictions = get_prediction(response)
        prediction_list.append(batch_predictions)
        label_list.append(batch_labels)

    prediction_list_int = [1 if prediction == 'Yes' else
                           0 if prediction == 'No'
                           else -1
                           for prediction in prediction_list]

    label_list_int = [1 if prediction == 'Yes' else
                      0 if prediction == 'No'
                      else -1
                      for prediction in label_list]

    # Remove illegal predictions
    illegal_indices = [i for i, val in enumerate(prediction_list_int) if val == -1]
    if illegal_indices:
        print(f'Illegal predictions found in {len(illegal_indices)} examples')
        print(f'Illegal Responses:')
        for index in illegal_indices[::-1]:
            print(f'Index: {index}, Response: {all_responses[index]}')
            prediction_list_int.pop(index)
            label_list_int.pop(index)

    report = classification_report(label_list_int, prediction_list_int)
    print(f'@@@ Classification report for {eval_dataset_name} dataset @@@')
    print(report)


# texts = [
#     "I told my wife she was drawing her eyebrows too high. She looked surprised.",
#     "The sky is blue and water is wet.",
#     "Why don’t skeletons fight each other? They don’t have the guts."
# ]
#
# def classify(sentence):
#     prompt = f"""Classify the following sentence as Funny or Not Funny.
#
# Sentence: "{sentence}"
#
# Answer with only one word: Funny or Not Funny."""
#     response = model.generate_content(prompt)
#     return response.text.strip()
#
# for t in texts:
#     label = classify(t)
#     print(f"Sentence: {t}\nPrediction: {label}\n")

a = 5
print(a)
