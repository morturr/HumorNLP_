import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import TrainingArguments
from trl import SFTTrainer  # or use Trainer if you prefer
from peft.utils import prepare_model_for_kbit_training

from FlanT5.data_loader import load_dataset

from sklearn.metrics import (
    classification_report,)

# Load PEFT config (from your previous LoRA fine-tuning)
peft_model_id = "morturr/Mistral-7B-v0.1-LOO_dadjokes-COMB_one_liners-comb1-seed28-NEW2025-01-27"
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Set up 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",  # or manually assign devices if needed
)

# Prepare the model for LoRA training (very important!)
model = prepare_model_for_kbit_training(model)

# Load LoRA adapter on top of the 4-bit quantized model
model = PeftModel.from_pretrained(model,
                                  peft_model_id,
                                  is_trainable=True
                                  )

# Check trainable params (optional but useful)
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

print_trainable_parameters(model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
# Change padding side to right for training
tokenizer.padding_side = "right"
assert tokenizer.padding_side == "right"


dataset = load_dataset('dadjokes', instruction_version=0,
                       train_size=100, add_instruction=True, with_val=False)
train_dataset = dataset['train']

training_args = TrainingArguments(
    output_dir="./continued-lora-finetune",
    per_device_train_batch_size=2,
    learning_rate=3e-4,
    num_train_epochs=2,
    save_total_limit=2,
    save_steps=100,
    seed=28,
    gradient_accumulation_steps=4,
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    dataset_text_field="instruction",
)

trainer.train()

pass

print('Now Evaluating')

pass



tokenizer.padding_side = "left"
assert tokenizer.padding_side == "left"
amazon_eval = load_dataset('amazon', instruction_version=0,
                                        test_percent=0.1, add_instruction=True, with_val=False,
                                        percent=0.1)

dadjokes_eval = load_dataset('dadjokes', instruction_version=0,
                                        test_percent=0.1, add_instruction=True, with_val=False,
                                        percent=0.1)

def remove_response(row):
    # output = 'Yes' if row['label'] == 1 else 'No'
    # row['instruction'] = row['instruction'] + output
    instruction = row['instruction']
    terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
    # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
    row['instruction'] = instruction[:instruction.index(terminator) + len(terminator)]
    return row

def get_response(row):
    # output = 'Yes' if row['label'] == 1 else 'No'
    # row['instruction'] = row['instruction'] + output
    # instruction = row['instruction']
    # terminator = '### Output:\n' if '### Output:\n' in instruction else '### Response:\n'
    # # row['instruction'] = instruction[:instruction.index('### Output: ') + len('### Output: ')]
    row['text label'] = 'Yes' if row['label'] == 1 else 'No'
    # row['instruction'] = instruction[instruction.index(terminator) + len(terminator):]
    return row

def get_prediction(response):
    if '### Response:\n' not in response:
        return 'Illegal'

    response_idx = response.index('### Response:\n') + len('### Response:\n')
    response = response[response_idx:]
    if 'Yes' in response and 'No' in response:
        return 'Illegal'
    if 'Yes' in response:
        return 'Yes'
    if 'No' in response:
        return 'No'
    return 'Illegal'

with torch.no_grad():
    # eval_dataset = dadjokes_eval
    eval_dataset = dataset
    split = 'train'

    # split = 'test'
    true_labels = eval_dataset[split].map(get_response)

    # Remove the response from the instruction (If there is one)
    eval_dataset[split] = eval_dataset[split].map(remove_response)

    true_labels = [true_labels[i]['text label'] for i in range(len(true_labels))]
    texts = [eval_dataset[split][i]['instruction'] for i in range(len(eval_dataset[split]))]

    inference_batch_size = 2
    all_responses = []
    prediction_list, label_list = [], []


    for i in range(0, len(texts), inference_batch_size):
        batch_texts = texts[i: i + inference_batch_size]
        batch_labels = true_labels[i: i + inference_batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to("cuda")
        outputs = trainer.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                      attention_mask=inputs["attention_mask"],
                                      max_new_tokens=5,
                                      pad_token_id=tokenizer.eos_token_id)

        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        all_responses.extend(responses)
        batch_predictions = [get_prediction(response) for response in responses]
        prediction_list.extend(batch_predictions)
        label_list.extend(batch_labels)

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
        for index in illegal_indices[::-1]:
            prediction_list_int.pop(index)
            label_list_int.pop(index)

    report = classification_report(label_list_int, prediction_list_int)
    print(report)