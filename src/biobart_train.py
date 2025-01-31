import os
import numpy as np
import pandas as pd
import evaluate
import nltk
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from sklearn.metrics import classification_report

# Download required NLTK resource
nltk.download("punkt")

# Define label mappings
LABELS = [
    'None', 'part_of', 'Physically_related_to', 'start', 'improve', 'experiences',
    'affects', 'Marker/Mechanism', 'treats', 'Associated_with', 'possible',
    'Location_of', 'Negative_correlation', 'Interacts_with', 'worsen', 'causes',
    'prevents', 'decrease', 'increase', 'presence', 'complicates', 'stop', 'reveals'
]
id2label = {idx: label for idx, label in enumerate(LABELS)}
label2id = {label: idx for idx, label in enumerate(LABELS)}

# Load dataset
dataset = load_dataset(
    'csv',
    data_files={
        'train': "MicrobioRel/train.csv",
        'validation': "MicrobioRel/dev.csv",
        'test': "MicrobioRel/test.csv"
    }
)

# Map labels to numerical IDs
def map_labels(example):
    example['label'] = label2id.get(example['label'], 0)  # Default to 'None'
    return example

dataset = dataset.map(map_labels)

# Load tokenizer and model
MODEL_ID = "GanjinZero/biobart-v2-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")

# Shuffle training data
dataset['train'] = dataset['train'].shuffle(seed=42)

# Convert dataset to DataFrame and back for consistency
for split in ['train', 'validation', 'test']:
    df = pd.DataFrame(dataset[split])
    df['label'] = df['label'].astype(str)
    dataset[split] = Dataset.from_pandas(df)

# Determine max sequence lengths
all_texts = concatenate_datasets([dataset[s] for s in ['train', 'validation', 'test']])
max_source_length = max(len(x) for x in tokenizer(all_texts['text'], truncation=True)['input_ids'])
max_target_length = max(len(x) for x in tokenizer(all_texts['label'], truncation=True)['input_ids'])
print(f"Max source length: {max_source_length}, Max target length: {max_target_length}")

# Tokenization function
def preprocess_function(example):
    inputs = tokenizer(example['text'], max_length=max_source_length, truncation=True, padding='max_length')
    labels = tokenizer(example['label'], max_length=max_target_length, truncation=True, padding='max_length')
    labels['input_ids'] = [l if l != tokenizer.pad_token_id else -100 for l in labels['input_ids']]
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'label'])

# Define metric
metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"{MODEL_ID.split('/')[-1]}-MicrobioRel-text-classification",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=False,
    learning_rate=1e-5,
    num_train_epochs=30,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate()

# Save tokenizer
tokenizer.save_pretrained(training_args.output_dir)

# Prediction on test set
samples = len(dataset['test'])
predictions, labels = [], []
progress_bar = tqdm(range(samples))

for i in range(samples):
    text = dataset['test']['text'][i]
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=512).to(model.device)
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4)
    predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    labels.append(dataset['test']['label'][i])
    progress_bar.update(1)

# Classification report
report = classification_report([str(l) for l in labels], predictions, zero_division=0)
print(report)



