import os
import uuid
import argparse

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score, accuracy_score, classification_report

from datasets import load_dataset, load_metric

import transformers
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, TrainingArguments, Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
print(transformers.__version__)

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("-m", "--model_name", help = "HuggingFace Hub model name")
args = vars(parser.parse_args())


corpus_name = "microbioRel"


dataset_base = load_dataset(
    'csv',
    data_files={
        'train': f"/home/microbiome_RE/datasets/masked_data/{corpus_name}/train.csv",
        'validation': f"/home/microbiome_RE/datasets/masked_data/{corpus_name}/dev.csv",
        'test': f"/home/microbiome_RE/datasets/masked_data/{corpus_name}/test.csv",
    }
)
dataset_base = dataset_base.rename_column("label", "labels")

dataset_train = dataset_base["train"]
print(len(dataset_train))

dataset_val = dataset_base["validation"]
print(len(dataset_val))

dataset_test = dataset_base["test"]
print(len(dataset_test))

metric = load_metric("accuracy")

labels_list = ['None', 'part_of', 'Physically_related_to', 'start', 'improve','experiences', 'affects', 'Marker/Mechanism', 'treats',
       'Associated_with', 'possible', 'Location_of',
       'Negative_correlation', 'Interacts_with', 'worsen', 'causes',
       'prevents', 'decrease', 'increase', 'presence', 'complicates',
       'stop', 'reveals']
print(labels_list)

num_labels = len(labels_list)
print("num_labels: ", num_labels)

batch_size = 4
EPOCHS = 30
model_checkpoint = str(args["model_name"])

id2label = {idx:label for idx, label in enumerate(labels_list)}
print(id2label)
label2id = {label:idx for idx, label in enumerate(labels_list)}
print(label2id)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, use_auth_token="")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    use_auth_token="",
)

model_name = model_checkpoint.split("/")[-1]

print("#"*50)
print(model.config.max_position_embeddings)
print("#"*50)


def preprocess_function(e):
    
    res = tokenizer(e["text"], truncation=True, max_length=512, padding="max_length")
    
    label = label2id[e["labels"]]
    res["labels"] = label

    return res

dataset_train = dataset_train.map(preprocess_function, batched=False)
dataset_train.set_format("torch")

dataset_val   = dataset_val.map(preprocess_function, batched=False)
dataset_val.set_format("torch")

dataset_test   = dataset_test.map(preprocess_function, batched=False)
dataset_test.set_format("torch")

output_dir = f"./models/Task_1-{model_name}-finetuned-{uuid.uuid4().hex}"

true_labels = list(dataset_test["labels"])


output_dir = f"./RE/models/Task_1-{model_name}-finetuned-{uuid.uuid4().hex}"

training_args = TrainingArguments(
    output_dir,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    # learning_rate=2e-5,
    learning_rate=1e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=1e-6,
    # weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
    greater_is_better=True,
    metric_for_best_model="f1_score",
    # metric_for_best_model="accuracy",
)

def toLogits(predictions):
    
    
    logit = torch.Tensor(predictions)
    y_pred = np.asarray(torch.argmax(logit, dim=-1))

    
    return y_pred

def label_metrics(predictions, labels):

    y_predd = toLogits(predictions)
    y_truee = labels

    f1 = f1_score(y_true=y_truee, y_pred=y_predd, average='micro')
    accuracy = accuracy_score(y_truee, y_predd)

    metrics = {'f1_score': f1, 'accuracy': accuracy}

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    result = label_metrics(
        predictions=preds, 
        labels=p.label_ids
    )
    return result

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

best_model_path = f"{output_dir}/models/best_model/"
trainer.save_model(best_model_path)
print(f"Best model is saved at : {best_model_path}")

print(trainer.evaluate())

# ------------------TEST ------------------

predictions, labels, _ = trainer.predict(dataset_test)

predictions = toLogits(predictions)
print("Predictions Logits:")
print(predictions)
print("labels:")
print(labels)

#metrics = label_metrics(predictions, true_labels)
#print(metrics)

cr = classification_report(true_labels, predictions, target_names=labels_list, digits=4)
print(cr)

# Save the classification report to a text file with model_name in the filename
classification_report_file = f"{output_dir}/classification_report_{model_name}.txt"

# Save the classification report to a text file
with open(classification_report_file, 'w') as f:
    f.write("Classification Report\n")
    f.write(cr)
    f.write("\n")  # Ensure there's a newline at the end
