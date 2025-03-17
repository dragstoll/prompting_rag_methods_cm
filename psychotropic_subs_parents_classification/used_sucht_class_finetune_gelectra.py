#!/usr/bin/env python
# coding: utf-8

# Activate the conda environment (commented out)
#!conda activate huggingface_env
#conda info
#ls

# Import necessary packages
import sys
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig 
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, AutoTokenizer, GPTJForCausalLM, GPTJConfig, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoConfig
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import trange
import random
from ml_things import plot_dict, plot_confusion_matrix, fix_text
import os

# Disable Weights & Biases logging
os.environ['WANDB_DISABLED'] = 'true'
pd.options.display.max_colwidth = 1000

from pathlib import Path    

# Get the current working directory
cwd = os.getcwd()

# Load sample data from Excel files
sample_path0 = os.path.expanduser('rbs_sucht_eltern_sampleTemp_4k_klassifiziert.xlsx')
sample_df0 = pd.read_excel(sample_path0, sheet_name='Sheet1')
sample_df0 = sample_df0.rename(columns={'validiert': 'code'})  # Rename 'validierung' column to 'code'
sample_df0 = sample_df0[['antwort', 'code']]
sample_df0 = sample_df0.rename(columns={'antwort': 'sentence', 'code': 'label'})
sample_df0 = sample_df0.dropna()
sample_df0 = sample_df0.reset_index(drop=True)

sample_path1 = os.path.expanduser('rbs_eltern4000-5000_sucht_codiert_validiert.xlsx')
sample_df1 = pd.read_excel(sample_path1, sheet_name='Sheet1')
sample_df1 = sample_df1.rename(columns={'validiert': 'code'})  # Rename 'validierung' column to 'code'
sample_df1 = sample_df1[['antwort', 'code']]
sample_df1 = sample_df1.rename(columns={'antwort': 'sentence', 'code': 'label'})
sample_df1 = sample_df1.dropna()
sample_df1 = sample_df1.reset_index(drop=True)

# Load another sample data file
sample_path = os.path.expanduser('synt_traintest_sucht.xlsx')
sample_df = pd.read_excel(sample_path, sheet_name='Sheet1')
sample_df = sample_df.dropna()
sample_df = sample_df.reset_index(drop=True)

# Rename columns and filter data
sample_df = sample_df.rename(columns={'validierung': 'code'})  # Rename 'validierung' column to 'code'
sample_df = sample_df[sample_df['code'] != -1]
sample_df = sample_df[['antwort', 'code']]
sample_df = sample_df.rename(columns={'antwort': 'sentence', 'code': 'label'})

# Concatenate the dataframes
sample_df = pd.concat([sample_df, sample_df0, sample_df1])
sample_df = sample_df.reset_index(drop=True)
sample_df['label'] = sample_df['label'].astype(int)

# Split data into training and testing sets
test_data_label_0 = sample_df[sample_df['label'] == 0].sample(n=25, random_state=42)
test_data_label_1 = sample_df[sample_df['label'] == 1].sample(n=25, random_state=42)
test_data = pd.concat([test_data_label_0, test_data_label_1])
train_data = sample_df.drop(test_data.index)

# Print label counts
print("Count values of label variable in test_data:")
print(test_data['label'].value_counts())
print("Count values of label variable in train_data:")
print(train_data['label'].value_counts())

# Drop NA values
train_data = train_data.dropna()
test_data = test_data.dropna()
train = train_data

# Define label strings
psych_eltern_string = ["SuchtErkankungen bei Eltern nicht erwähnt", "SuchtErkankungen bei Eltern erwähnt"]

# Load model and tokenizer
model_name = "deepset/gelectra-large"
print("Model name:", model_name)

id2label = {
    "0": "SuchtErkankungen bei Eltern nicht erwähnt",
    "1": "SuchtErkankungen bei Eltern erwähnt",    
}

label2id = {
    "SuchtErkankungen bei Eltern nicht erwähnt": 0,
    "SuchtErkankungen bei Eltern erwähnt": 1,    
}

model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, 
                                          id2label=id2label, 
                                          label2id=label2id, 
                                          hidden_dropout_prob=0.1, 
                                          attention_probs_dropout_prob=0.1,
                                          summary_last_dropout=0.1,
                                          output_attentions=False, 
                                          return_dict=False)

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, use_fast=True, eos_token='###', pad_token='[PAD]')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# Convert data to Hugging Face datasets
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(test_data)
test_dataset = Dataset.from_pandas(test_data)

# Remove unnecessary columns
test_dataset = test_dataset.remove_columns(["__index_level_0__"])
valid_dataset = valid_dataset.remove_columns(["__index_level_0__"])
train_dataset = train_dataset.remove_columns(["__index_level_0__"])

# Create dataset dictionary
from datasets import DatasetDict
datasets = DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset,
})

# Tokenize data
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True, max_length=512, padding=True, return_tensors="pt")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Define training arguments
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# Define metric for evaluation
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Remove existing results folder
import shutil
folder_path = "results_gelectra_sucht_finetune"
shutil.rmtree(folder_path)

# Define detailed training arguments
training_args = TrainingArguments(output_dir='results_gelectra_sucht_finetune', 
                                  num_train_epochs=12, 
                                  save_total_limit=12,
                                  save_strategy="epoch", 
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=4, 
                                  per_device_eval_batch_size=4, 
                                  gradient_accumulation_steps=1,
                                  gradient_checkpointing=True, 
                                  optim="paged_adamw_32bit",
                                  warmup_ratio=0.1,
                                  weight_decay=0.001, 
                                  logging_dir='logs', 
                                  learning_rate=2e-05,
                                  fp16=False,
                                  lr_scheduler_type="cosine")

# Initialize Trainer
trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=tokenized_datasets['train'], 
                  eval_dataset=tokenized_datasets['valid'], 
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer,
                  data_collator=data_collator)

# Start training
trainer.train()

# Evaluate model
predictions = trainer.predict(tokenized_datasets['valid'])
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(np.argmax(predictions[0], axis=-1), tokenized_datasets['valid']['labels']))

# Create evaluation report
evaluation_report = classification_report(tokenized_datasets['valid']['labels'], np.argmax(predictions[0], axis=-1), target_names=psych_eltern_string)
print(evaluation_report)

# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels, title2="Normalized confusion matrix", path='plot_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=True, xticks_rotation='vertical')
    plt.title(title2)
    plt.grid(False)
    plt.show()
    plt.savefig(path)

eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=1, collate_fn=data_collator)

# Move model to GPU if available
model.to(device)
model.eval()

predictions_sum = []
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        preds, logits = model(**batch)
    predictions = torch.argmax(logits, dim=-1)
    predictions_sum += predictions

predictions_sum = torch.Tensor(predictions_sum)
predictions_sum.cpu()
plot_confusion_matrix(predictions_sum, tokenized_datasets["valid"]["labels"], psych_eltern_string, title2="Normalized confusion matrix\n Validation Dataset", path='validation_plot_confusion_matrix.png')

# Save model and tokenizer
model.save_pretrained("sucht_class_gelectra_rag2", safe_serialization=True)
tokenizer.save_pretrained("sucht_class_gelectra_rag2")
