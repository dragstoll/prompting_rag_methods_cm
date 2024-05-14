#!/usr/bin/env python
# coding: utf-8


#!conda activate huggingface_env
#conda info
#ls
# import packages
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
os.environ['WANDB_DISABLED'] = 'true'
pd.options.display.max_colwidth = 1000

from pathlib import Path    
import os





cwd = os.getcwd()
# sample_path = os.path.expanduser('rbs_ecan_klient_sample_antwort_CommandRdirekt1-500_annotiert.xlsx')
# sample_path = os.path.expanduser('answerTemp_annotiert.xlsx')
# sample_df0 = pd.read_excel(sample_path, sheet_name='Sheet1')
# sample_df0 = sample_df0.dropna(subset=['code'])


# sample_df0 = sample_df0[['antwort', 'code']]
# sample_df0 = sample_df0.rename(columns={'antwort': 'sentence', 'code': 'label'})

sample_path = os.path.expanduser('rbs_ecan_klient_sample_antwort_recursFaissRerankBGEM22_1-500_annotiert.xlsx')
sample_df = pd.read_excel(sample_path, sheet_name='Sheet1')
sample_df = sample_df.dropna(subset=['code'])


sample_df = sample_df[['antwort', 'code']]
sample_df = sample_df.rename(columns={'antwort': 'sentence', 'code': 'label'})

#concat the two dataframes
# sample_df = pd.concat([sample_df, sample_df0])
sample_df = sample_df.reset_index(drop=True)





# Select 25 rows with label equal to 0
test_data_label_0 = sample_df[sample_df['label'] == 0].sample(n=50, random_state=42)

# Select 25 rows with label equal to 1
test_data_label_1 = sample_df[sample_df['label'] == 1].sample(n=50, random_state=42)

# Select 25 rows with label equal to 2
# test_data_label_2 = sample_df[sample_df['label'] == 2].sample(n=21, random_state=42)

# Concatenate the selected rows
test_data = pd.concat([test_data_label_0, test_data_label_1, 
                    #    test_data_label_2
                       ])

# Save the rest of the rows in train_data
train_data = sample_df.drop(test_data.index)

test_data['label'] = test_data['label'].astype(int)
train_data['label'] = train_data['label'].astype(int)


print("Count values of label variable in test_data:")
print(test_data['label'].value_counts())

print("Count values of label variable in train_data:")
print(train_data['label'].value_counts())
# train_data_label_0 = train_data[train_data['label'] == 0].sample(n=85, random_state=42)
# train_data_label_1 = train_data[train_data['label'] == 1]

# train_data = pd.concat([train_data_label_0, train_data_label_1])

# train_data_label_1_oversampled = train_data[train_data['label'] == 1].sample(n=215, replace=True, random_state=42)
# train_data = pd.concat([train_data_label_0, train_data_label_1_oversampled])




print("Count values of label variable in train_data:")
print(train_data['label'].value_counts())
#drop na values
train_data=train_data.dropna()
test_data=test_data.dropna()
train=train_data



from sklearn.utils import shuffle

psych_eltern_string = [ "psychische Misshandlung beim Kind nicht erwähnt", "psychische Misshandlung beim Kind erwähnt"]
# \nA: sich verbessernde Entwicklung \nB: gleichbleibende, sich nicht verbessernde Entwicklung oder uneinheitliche, unklare Entwicklung  \nC: sich verschlechternde Entwicklung


from transformers import AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig
import aiohttp
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING=1
print(device)

from datasets import load_dataset

import aiohttp
# model_name = "microsoft/deberta-v2-xxlarge"
# model_name = "microsoft/mdeberta-v3-base"
model_name ="deepset/gelectra-large"
# model_name = "german-nlp-group/electra-base-german-uncased"
#print model name
print("Model name:", model_name)

# model_name_clean=model_name.replace("/","_")
# #create a variable with date and time stamp 
# import datetime
# now = datetime.datetime.now()
# date_time_stamp = now.strftime("%Y%m%d_%H%M")
# import argparse, shutil, logging
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name_save', type=str, default=model_name_clean)

# args = parser.parse_args()
# model_name_save = args.model_name_save
# model_date_time_stamp = model_name_save+"_"+date_time_stamp

# model_name= "ZurichNLP/swissbert"
# model_name = "GroNLP/mdebertav3-subjectivity-german"
# Get model configuration.
# "A: sich verbessernde Entwicklung", 
# "B: gleichbleibende, sich nicht verbessernde Entwicklung oder uneinheitliche, unklare Entwicklung", 
# "C: sich verschlechternde Entwicklung"

id2label={
    
    "0": "psychische Misshandlung beim Kind nicht erwähnt",
    "1": "psychische Misshandlung beim Kind erwähnt",    
  }

label2id={
    
    "psychische Misshandlung beim Kind nicht erwähnt": 0,
    "psychische Misshandlung beim Kind erwähnt": 1,    
  }, 



model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, 
# num_labels=2, 
id2label=id2label, 
label2id=label2id, 

hidden_dropout_prob=0.2, 
attention_probs_dropout_prob=0.2,
summary_last_dropout=0.2,
# cls_drop_out=0.1,
output_attentions=False, 
 return_dict=False, 
)
# import accelerate

model = AutoModelForSequenceClassification.from_pretrained(model_name,  config=model_config, 
                                                            #   device_map='auto',
#                                                            num_labels=2,
                                                               
#                                                            id2label=id2label, 
                                                            # label2id=label2id, 
                                                            # ignore_mismatched_sizes=True,
                                                            # use_safetensors=True,
                                                                )
# .half().to(device)



# model.set_default_language("de_CH")

# #print model 
# print(model)
# # sys.exit()
# #freeze everything except classifier and last two transformer layers
# for param in model.base_model.parameters():
#     param.requires_grad = False
# for param in model.base_model.encoder.layer[-1].parameters():
#     param.requires_grad = True
# for param in model.base_model.encoder.layer[-2].parameters():
#     param.requires_grad = True
# for param in model.classifier.parameters():
#     param.requires_grad = True



tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          do_lower_case = True,
                                         use_fast=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

from datasets import Dataset
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case = True)




#mit overrepresentation
# train_dataset = Dataset.from_pandas(train_over)
#mit underrepresentation
# train_dataset = Dataset.from_pandas(train_under)
#mit overunderrepresentation
# train_dataset = Dataset.from_pandas(train_overunder)
# #ohne overrepr
# # train_dataset = Dataset.from_pandas(train)
# valid_dataset = Dataset.from_pandas(valid)

# test_dataset = Dataset.from_pandas(valid)

train_dataset = Dataset.from_pandas(train_data)
#ohne overrepr
# train_dataset = Dataset.from_pandas(train)
valid_dataset = Dataset.from_pandas(test_data)

test_dataset = Dataset.from_pandas(test_data)

test_dataset = test_dataset.remove_columns([ "__index_level_0__" ])

valid_dataset = valid_dataset.remove_columns([ "__index_level_0__"] )
train_dataset = train_dataset.remove_columns([  "__index_level_0__"] )
# test_dataset = test_dataset.remove_columns(["Unnamed: 0.1", "Unnamed: 0", "__index_level_0__" ])

# valid_dataset = valid_dataset.remove_columns(["Unnamed: 0.1", "Unnamed: 0", "__index_level_0__"] )
# train_dataset = train_dataset.remove_columns(["Unnamed: 0.1", "Unnamed: 0", "__index_level_0__"] )

from datasets import DatasetDict
datasets = DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset,
    # "test": test_dataset
    })
print(datasets)
                                          


def tokenize_function(example):
    return tokenizer(example["sentence"] ,  truncation=True, max_length=512, padding=True, return_tensors="pt")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# from datasets import load_metric

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
# #     print(logits)
# #     print(logits[0])
# #     print(labels)
# #     print(logits[1])
#     predictions = np.argmax(logits[0], axis = -1)
#     # print(predictions)
# #     print(labels)
#     return metric.compute(predictions=predictions, references=labels)

import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print('Loading configuration...')




training_args = TrainingArguments(output_dir='results_gelectra_ecan_finetune', 
                                  num_train_epochs=4, 
                                #   logging_steps=100, 
                                #   save_steps=100,
                                 load_best_model_at_end=True, 
                                 save_total_limit=2,
                                 save_strategy="epoch", evaluation_strategy="epoch",
                                # save_strategy="steps", evaluation_strategy="steps",
#                                      save_steps=10, 
#                                     eval_steps=10,
                                    eval_accumulation_steps=1,
                                 per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=1,gradient_checkpointing=True, 
#                                  optim="adafactor",
                                  # optim="adamw_hf",
                                optim="paged_adamw_32bit",
                                #  warmup_steps=100,
                                  warmup_ratio=0.1,
                                     weight_decay=0.2, 
                                  logging_dir='logs', 
                                  learning_rate=2e-05,
                                 fp16 = False,
                                 lr_scheduler_type="cosine",)

# start training
trainer =Trainer(model=model, args=training_args, 
                 train_dataset=tokenized_datasets['train'],                  
                 eval_dataset=tokenized_datasets['valid'],  
                 compute_metrics=compute_metrics,
                 tokenizer=tokenizer,
                 data_collator=data_collator,)
# 
        # data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
        #                             'attention_mask': torch.stack([f[1] for f in data]),
        #                             'labels': torch.stack([f[0] for f in data])})

        
trainer.train()

print(os.getcwd())
# trainer.train("results_electra_gewalt_klassen/checkpoint-1440")



predictions = trainer.predict(tokenized_datasets['valid'])

# print(predictions[2])
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(np.argmax(predictions[0], axis=-1), tokenized_datasets['valid']['labels']))
# predicted = np.argmax(predictions[0][0], axis=-1)
# Create the evaluation report.

evaluation_report = classification_report(tokenized_datasets['valid']['labels'], np.argmax(predictions[0], axis=-1),  target_names=psych_eltern_string)
# evaluation_report = classification_report(tokenized_datasets['test']['label'], np.argmax(predictions[0][0], axis=-1), labels=tokenized_datasets['test']['label'], target_names=tokenized_datasets['test']['label'])
# Show the evaluation report.
print(evaluation_report)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels, title2="Normalized confusion matrix", path='plot_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_preds , normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=True, xticks_rotation='vertical') 
    # plt.title("Normalized confusion matrix")
    plt.title(title2)
    plt.grid(False)
    plt.show()
    plt.savefig(path)


eval_dataloader = DataLoader(
    tokenized_datasets["valid"], batch_size=1, collate_fn=data_collator
)


#mode mocel to gpu
model.to(device)

model.eval()
predictions_sum=[]
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        preds, logits = model(**batch)
    # print(outputs.shape)    
    #create logits but AttributeError: 'tuple' object has no attribute 'logits'
    # logits = outputs[][]
    
    
    
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

#print accuracy score
print(metric.compute())

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        #generate outputs
        preds, logits = model(**batch)
        

    #create logits
    # logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    predictions_sum+=predictions
    # print(logits)
    # print(predictions)
    # predictions.add_batch(predictions=predictions, references=batch["labels"])
    
    # precision_metric.add_batch(predictions=predictions, references=batch["labels"])

# hide_output
# preds_output = model.predict(tokenized_datasets["valid"])
# preds_output.metrics





predictions_sum = torch.Tensor(predictions_sum)
predictions_sum.cpu()
# len(predictions_sum)
plot_confusion_matrix(predictions_sum, tokenized_datasets["valid"]["labels"], psych_eltern_string, title2="Normalized confusion matrix\n Validation Dataset", path='validation_plot_confusion_matrix.png')

# model.save_pretrained("alk_class_gelectra")
model.save_pretrained("ecan_klient_class_gelectraMx22", safe_serialization=True)
tokenizer.save_pretrained("ecan_klient_class_gelectraMx22")



# model.push_to_hub("gelectra_20231020", use_auth_token=True, private=True, safe_serialization=True)
# print("Model pushed to hub")

#save tokenizer
# tokenizer.save_pretrained("alk_prompt_class_gelectra", safe_serialization=True)
# tokenizer.push_to_hub("gelectra_20231020", use_auth_token=True, private=True, safe_serialization=True)