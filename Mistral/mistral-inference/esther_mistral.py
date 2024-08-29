from mistral_inference.model import Transformer

from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage

from mistral_common.protocol.instruct.request import ChatCompletionRequest

from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

import argparse

import pandas as pd

from bs4 import BeautifulSoup

import seaborn as sn

from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,

                             accuracy_score, f1_score, precision_score, recall_score,

                             precision_recall_fscore_support)

import matplotlib.pyplot as plt

import sys

import time

sys.path.insert(0, "/cs/labs/oabend/esthersh/finetune_with_unsloth/prompting/prompts")

from prompts import *

"""

https://github.com/mistralai/mistral-common/releases 

"""

OUT_DIR = "/cs/labs/oabend/esthersh/mistral-finetune"


def format_prompt(row, only_rc=False):
    if only_rc and not row["religious_content"]:
        return ""

    user_prompt_template = ('Below is an instruction that describes a task, paired with an input sample and reasoning '

                            'that provide further context.\nWrite a response that appropriately completes the '

                            'request.\n<instruction>\n{instruction}\n</instruction>\n'

                            '<input>\n{input}\n</input>')

    instruction = SYS_PROMPTS.get(prompt_id)

    user_prompt = user_prompt_template.format(instruction=instruction,

                                              input=row["text"])

    return user_prompt


def predict(prompt: str, tokenizer: MistralTokenizer, model: Transformer):
    if prompt == "":
        return ""

    completion_request = ChatCompletionRequest(

        messages=[SystemMessage(content=GENERIC_SYS_PROMPT), UserMessage(content=prompt)])

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model,

                             max_tokens=1024,

                             temperature=0.0,

                             eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

    if len(out_tokens) > 0:

        return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    else:

        return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens)


def clean(text):
    if "ACTIVE" in text:
        return "ACTIVE"

    if "INACTIVE" in text:
        return "INACTIVE"

    if "POSITIVE" in text:
        return "POSITIVE"

    if "NEGATIVE" in text:
        return "NEGATIVE"

    if "AMBIGUOUS" in text:
        return "AMBIGUOUS"

    return text


def get_class_body(text):
    # phrase = text.split("<classification>")[-1].split()[0]  #.apply(lambda x: extract_classification_body(x)).unique()

    # pattern = re.compile(r"\w+")

    # return pattern.search(phrase).group(0)

    res = text.split("<classification>")[-1].split()[0]

    return clean(res.split("</classification>")[0])


def plot_confusion_matrix(predictions_df):
    """

    TODO - adapt this to the right script.....!!!

    :param predictions_df:

    :return:

    """

    print("\n~~ Predictions head ~~\n")

    print("gold label: ", predictions_df["labels"].head(2))

    print("pred label: ", predictions_df["pred_label"].head(2))

    print("Unique gold labels: ", predictions_df["labels"].unique())

    print("Unique pred labels: ", predictions_df["pred_label"].unique())

    confusion_m = confusion_matrix(y_true=predictions_df["labels"],

                                   y_pred=predictions_df["pred_label"],

                                   labels=predictions_df["labels"].unique())

    plt.figure()

    fig = sn.heatmap(confusion_m,

                     xticklabels=predictions_df["labels"].unique(),

                     yticklabels=predictions_df["labels"].unique(),

                     annot=True,

                     cmap=sn.color_palette("crest", as_cmap=True))

    # Set the axis labels and title

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.title(f"Mistral FT model, confusion matrix\n"

              f"for {predictions_df.shape[0]} samples")

    # Add legends for the heatmap

    # bottom, top = plt.ylim()

    # plt.ylim(bottom + 0.5, top - 0.5)

    plt.xticks(rotation=0)

    plt.yticks(rotation=0)

    plt.show()

    return confusion_m


def analyze_results(data: pd.DataFrame, model_name):
    """

    prints the accuracy and f1 of the model

    prints the classification report

    plots the confusion matrix

    saves the confusion matrix

    :param data:

    :return:

    """

    print(classification_report(data.labels,

                                data.pred_label, ))

    conf_matrix = plot_confusion_matrix(data)

    # print("Accuracy: {:.1f}".format(results["eval_accuracy"] * 100))

    # confusion matrix of predictions etc//

    plt.savefig(f'{OUT_DIR}/confusion_matrices/{model_name}.png')

    plt.clf()

    print(f"\n~~ {conf_matrix} ~~\n")

    print(f"\n~~ Classification Report ~~\n")

    text_report = classification_report(y_true=data.labels,

                                        y_pred=data.pred_label)

    # f1_macro = f1_score(y_true=data.gold_label,

    #                     y_pred=data.pred_label,

    #                     average='macro')

    print(text_report)

    with open(f"{OUT_DIR}/results/{model_name}.txt", "w") as f:
        f.write(f"model: {model_name} \n")

        f.write(str(text_report))


if __name__ == '__main__':

    """

    produces the output of the model given validation/test data

    saves the predictions as a csv file

    prints the accuracy and f1 of the model

    prints the classification report

    plots the confusion matrix

    saves the confusion matrix



    args:

    --data_path: str, path to the data file

    --model_path: str, path to the model file

    --tokenizer_path: str, path to the tokenizer file

    --adapters_path: str, path to the adapter

    --prompt_id: int, the id of the prompt to use

    --validating: bool, if True, the model is validated on the data



    example:

    python inference.py --data_path "/cs/labs/oabend/esthersh/finetune_with_unsloth/data/belief_eval_data_2024-07-02.csv"

     --model_path "/cs/labs/oabend/esthersh/mistral-finetune/mistral_models/7B"

     --tokenizer_path "/cs/labs/oabend/esthersh/mistral-finetune/mistral_models/7B/tokenizer.model.v3"

     --adapters_path "/cs/labs/oabend/esthersh/mistral-finetune/ft_mistral7B_500_steps_seed_1_no_eval/checkpoints/checkpoint_000100/consolidated/lora.safetensors"

     --adapter_name "ft_mistral7B_500_steps_seed_1_no_eval"

     --checkpoint_no 100

     --prompt_id = "TF_FT_VB0"

    """

    print("\n ~ Parsing args... ~ \n")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)

    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--tokenizer_path', type=str, required=True)

    parser.add_argument('--adapters_path', type=str, required=False)

    parser.add_argument('--prompt_id', type=str, required=True)

    parser.add_argument('--adapter_name', type=str, required=False)

    parser.add_argument('--checkpoint_no', type=int, required=False)

    parser.add_argument('--validating', action='store_true')

    parser.add_argument('--only_religious_content', action='store_true')

    args = parser.parse_args()

    zero_shot = True if args.adapters_path is None else False

    prompt_id = args.prompt_id

    print("\n ~ Done parsing args ~ \n")

    if not zero_shot:

        model_name = f"{args.model_path.split('/')[-1]}_{args.adapter_name}_{args.checkpoint_no}"

    else:

        model_name = f"{args.model_path.split('/')[-1]}_zero_shot"

    # 7B tokentizer and model

    print("\n ~ Loading Model and tokenizer ~ \n")

    if args.tokenizer_path is None:

        tokenizer = MistralTokenizer.v3(is_tekken=True)

        # tokenizer = MistralTokenizer.v3(is_tekken=True)

        tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.KEEP  # or SpecialTokenPolicy.IGNORE



    else:

        tokenizer = MistralTokenizer.from_file(args.tokenizer_path)

    model = Transformer.from_folder(args.model_path)

    if not zero_shot:
        model.load_lora(args.adapters_path)

    print("\n ~ Model and tokenizer loaded ~ \n")

    input_data = pd.read_csv(args.data_path, index_col=0)

    input_data["prompt"] = input_data.apply(format_prompt,

                                            only_rc=args.only_religious_content,

                                            axis=1)

    print("\n ~ Predicting... ~ \n")

    # time the prediction

    start_time = time.time()

    input_data['prediction'] = input_data['prompt'].apply(lambda x: predict(x, tokenizer, model))

    end_time = time.time()

    print(f"Prediction of {input_data[input_data.prompt != ''].shape[0]} samples took {end_time - start_time} seconds")

    # convert to hours

    print(f"Prediction took {(end_time - start_time) / 3600} hours")

    print("\n ~ Done predicting ~ \n")

    out_name = f"{model_name}_{input_data.shape[0]}_samples"

    if args.only_religious_content:
        out_name += "_rc"

    input_data.to_csv(f"{OUT_DIR}/predictions/{out_name}.csv")

    # TODO - debug this

    input_data['pred_label'] = input_data['prediction'].apply(

        lambda x: BeautifulSoup(x, 'html.parser').find('classification').text.strip()

        if x != "" else "")

    if args.validating:
        out_name += "_valid"

    input_data.to_csv(f"{OUT_DIR}/predictions/{out_name}.csv")

    if args.validating:
        print("Validating...")

        analyze_results(input_data, model_name=model_name)

    # input_data['pred_label'] = input_data['prediction'].apply(get_class_body)

    print("Done!")