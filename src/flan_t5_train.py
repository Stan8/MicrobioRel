import os
import random
import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download("punkt")

# Define labels and mappings
LABELS = [
    'None', 'part_of', 'Physically_related_to', 'start', 'improve', 'experiences', 'affects',
    'Marker/Mechanism', 'treats', 'Associated_with', 'possible', 'Location_of', 'Negative_correlation',
    'Interacts_with', 'worsen', 'causes', 'prevents', 'decrease', 'increase', 'presence', 'complicates',
    'stop', 'reveals'
]
id2label = {idx: label for idx, label in enumerate(LABELS)}
label2id = {label: idx for idx, label in enumerate(LABELS)}

# Load dataset
dataset_base = load_dataset("csv", data_files={
    'train': "MicrobioRel/train.csv",
    'validation': "MicrobioRel/dev.csv",
    'test': "MicrobioRel/test.csv"
})

# Replace NaN values with 'None'
def replace_nan_with_none(example):
    return {key: (value if pd.notna(value) else 'None') for key, value in example.items()}

dataset_base = dataset_base.map(replace_nan_with_none)

# Convert labels to IDs
def map_labels_to_ids(example):
    example['label'] = label2id[example['label']]
    return example

dataset = dataset_base.map(map_labels_to_ids)

# Load tokenizer and model
MODEL_ID = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# Shuffle training data
dataset['train'] = dataset['train'].shuffle(seed=42)

# Convert dataset to Pandas DataFrame for processing
def to_dataframe(split):
    df = pd.DataFrame(dataset[split])
    df['label'] = df['label'].astype(str)
    return Dataset.from_pandas(df)

dataset = {split: to_dataframe(split) for split in ['train', 'validation', 'test']}

# Determine max sequence lengths
concatenated_data = concatenate_datasets([dataset[split] for split in ['train', 'validation', 'test']])

max_source_length = max(len(x) for x in concatenated_data.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)["input_ids"])
max_target_length = max(len(x) for x in concatenated_data.map(lambda x: tokenizer(x["label"], truncation=True), batched=True)["input_ids"])

print(f"Max source length: {max_source_length}")
print(f"Max target length: {max_target_length}")

# Preprocessing function
def preprocess_function(sample, padding="max_length"):
    model_inputs = tokenizer(sample["text"], max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["label"], max_length=max_target_length, padding=padding, truncation=True)
    
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = {split: dataset[split].map(preprocess_function, batched=True, remove_columns=['text', 'label']) for split in ['train', 'validation', 'test']}

# Load evaluation metric
metric = evaluate.load("f1")

def postprocess_text(preds, labels):
    preds = ["\n".join(sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(sent_tokenize(label.strip())) for label in labels]
    return preds, labels

# Compute evaluation metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds[0] if isinstance(preds, tuple) else preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
    return result

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="flan-t5-MicrobioRel-text-classification",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-5,
    num_train_epochs=30,
    logging_dir="logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train and evaluate model
trainer.train()
trainer.evaluate()

# Save tokenizer and model
tokenizer.save_pretrained(training_args.output_dir)

# Generate predictions
def generate_predictions():
    test_data = dataset['test']
    predictions, true_labels = [], []
    progress_bar = tqdm(range(len(test_data)))
    
    for i in range(len(test_data)):
        text = test_data['text'][i]
        inputs = tokenizer(text, padding='max_length', max_length=512, return_tensors='pt').to(model.device)
        output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
        
        predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
        true_labels.append(test_data['label'][i])
        progress_bar.update(1)
    
    print(classification_report([str(label) for label in true_labels], predictions, zero_division=0))

generate_predictions()