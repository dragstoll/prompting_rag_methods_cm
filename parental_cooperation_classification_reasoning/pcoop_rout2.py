# !pip install git+https://github.com/huggingface/transformers -q peft  accelerate bitsandbytes safetensors sentencepiece
# !pip install -U pandas

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# BitsAndBytesConfig
import pandas as pd
import sys

import argparse, shutil, logging
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,)
parser.add_argument('--target1', type=str,)
parser.add_argument('--target2', type=str,)
parser.add_argument('--iter', type=str,)

args = parser.parse_args()
source = args.source 
target1 = args.target1
target2 = args.target2
iter = args.iter


rbs_sample = pd.read_pickle(iter+'rbs_rout1_temp1.pkl')

from transformers import AutoTokenizer
model_name = "unsloth/Qwen3-32B-unsloth-bnb-4bit"

# Define stop token ids
stop_token_ids = [0]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import TextGenerationPipeline

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16
)
   
pipe = transformers.pipeline(
  "text-generation",
  model=model_name,
  model_kwargs={"torch_dtype": torch.float16, 
                # "load_in_4bit": True,
                 "quantization_config": bnb_config, 
                 "device_map": "cuda:0", 
                #  "attn_implementation": "flash_attention_2",
                 },
  max_new_tokens=50, do_sample=True, temperature=0.7, top_k=20, top_p=0.8, min_p=0,  pad_token_id=tokenizer.eos_token_id, return_full_text=False, 
 
)

# Generate prompts using the template
def format_prompt(text):
    messages = [
        {"role": "system", "content": " Anweisung: Interpretiere nicht, spekuliere nicht, was sein könnte, sondern beantworte die Fragen anhand der explizit erwähnten Hinweise."},
        {"role": "user", "content": f""" 
        Du hast nur 50 Tokens zur Verfügung, um die Antwort zu generieren.
Du darfst NUR! diese Outputs generieren!: 
{{"parental_coop_dad": "1"}}
{{"parental_coop_dad": "2"}}
{{"parental_coop_dad": "3"}}

Task:
Extrahiere aus dieser Antwort die Antwortkategorie im JSON-Format.
Wenn die Antwort 'a: mangelnde Kooperationsbereitschaft' ist dann {{"parental_coop_dad": "1"}},
Wenn die Antwort 'b: Kooperationsbereitschaft vorhanden, hat sich eingestellt' ist dann {{"parental_coop_dad": "2"}},
Wenn die Antwort 'c: keine Hinweise' ist dann {{"parental_coop_dad": "3"}}.


\nAntwort:\n
{text}


        """ }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,)


# Falls die es Hinweise für sowohl fehlende Kooperationsbereitschaft beim einem Elternteil gibt, als auch für vorhandene, sich einstellende Kooperationsbereitschaft gibt, dann beurteile die Kooperationsbereitschaft als vorhanden.

rbs_sample['father_antwort2'] = rbs_sample['father_antwort_content'].apply(format_prompt)
from datasets import Dataset
dataset = Dataset.from_pandas(rbs_sample)
from transformers.pipelines.pt_utils import KeyDataset

out_list = []
for out in pipe(KeyDataset(dataset, "father_antwort2"), ):    
    # print(out)
    out_list.append(out)
    
rbs_sample["father_antwort_dict2"]=out_list.copy()
rbs_sample["father_antwort_json"]=rbs_sample["father_antwort_dict2"].apply(lambda x: x[0]['generated_text'])
# rbs_sample["antwort_json"]

# extract from answer the start and end date this is a json format "{"start": "2014-02-01", "end": "2016-01-31"}"
import json
import re
def extract_kategories(answer):
  try:
    # Remove any triple backticks
    answer = re.sub(r'```.*?```', lambda m: m.group(0).replace('```', ''), answer, flags=re.DOTALL)
    # Find JSON between braces
    start = answer.find('{')
    end = answer.rfind('}')
    if start == -1 or end == -1:
      return None
    snippet = answer[start:end+1].replace('\\_', '_')
    # Try JSON parsing
    try:
      data = json.loads(snippet)
      if "parental_coop_dad" in data:
        return data["parental_coop_dad"]
    except:
      pass
    # Fallback to regex
    match = re.search(r'"parental_coop_dad"\s*:\s*"(.*?)"', answer)
    return match.group(1) if match else None
  except:
    return None

# Apply the function to the 'antwort' column
rbs_sample['kategorie_num_dad'] = rbs_sample['father_antwort_json'].apply(extract_kategories).astype(int)
# rbs_sample['kategorie_num_2try_rag_dad'] = rbs_sample['kategorie_num_2try_rag_dad'].replace({1: 2, 2: 1})




import numpy as np
def make_dichot(df, col, suffix="_dichot"):
    # Combine values 1 and 3 into 2, and keep 2 as 1; anything else -> NaN
    new_col = f"{col}{suffix}"
    df[new_col] = np.where(df[col].isin([2, 3]), 2,
                           np.where(df[col] == 1, 1, np.nan))
    return df

# Apply to requested variables
for col in [    
    'kategorie_num_dad',
]:
    rbs_sample = make_dichot(rbs_sample, col)




def format_prompt(text):
    messages = [
        {"role": "system", "content": " Anweisung: Interpretiere nicht, spekuliere nicht, was sein könnte, sondern beantworte die Fragen anhand der explizit erwähnten Hinweise."},
        {"role": "user", "content": f""" 
        Du hast nur 50 Tokens zur Verfügung, um die Antwort zu generieren.
Du darfst NUR! diese Outputs generieren!: 
{{"parental_coop_mom": "1"}}
{{"parental_coop_mom": "2"}}
{{"parental_coop_mom": "3"}}

Task:
Extrahiere aus dieser Antwort die Antwortkategorie im JSON-Format.
Wenn die Antwort 'a: mangelnde Kooperationsbereitschaft' ist dann {{"parental_coop_mom": "1"}},
Wenn die Antwort 'b: Kooperationsbereitschaft vorhanden, hat sich eingestellt' ist dann {{"parental_coop_mom": "2"}},
Wenn die Antwort 'c: keine Hinweise' ist dann {{"parental_coop_mom": "3"}}.

\nAntwort:\n
{text}


        """ }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,)


# Falls die es Hinweise für sowohl fehlende Kooperationsbereitschaft beim einem Elternteil gibt, als auch für vorhandene, sich einstellende Kooperationsbereitschaft gibt, dann beurteile die Kooperationsbereitschaft als vorhanden.

rbs_sample['mother_antwort2'] = rbs_sample['mother_antwort_content'].apply(format_prompt)


dataset = Dataset.from_pandas(rbs_sample)





out_list = []
for out in pipe(KeyDataset(dataset, "mother_antwort2"), ):    
    # print(out)
    out_list.append(out)
    
rbs_sample["mother_antwort_dict2"]=out_list.copy()
rbs_sample["mother_antwort_json"]=rbs_sample["mother_antwort_dict2"].apply(lambda x: x[0]['generated_text'])
# rbs_sample["antwort_json"]

# extract from answer the start and end date this is a json format "{"start": "2014-02-01", "end": "2016-01-31"}"
import json
import re
def extract_kategories(answer):
  try:
    # Remove any triple backticks
    answer = re.sub(r'```.*?```', lambda m: m.group(0).replace('```', ''), answer, flags=re.DOTALL)
    # Find JSON between braces
    start = answer.find('{')
    end = answer.rfind('}')
    if start == -1 or end == -1:
      return None
    snippet = answer[start:end+1].replace('\\_', '_')
    # Try JSON parsing
    try:
      data = json.loads(snippet)
      if "parental_coop_mom" in data:
        return data["parental_coop_mom"]
    except:
      pass
    # Fallback to regex
    match = re.search(r'"parental_coop_mom"\s*:\s*"(.*?)"', answer)
    return match.group(1) if match else None
  except:
    return None

# Apply the function to the 'antwort' column
rbs_sample['kategorie_num_mom'] = rbs_sample['mother_antwort_json'].apply(extract_kategories).astype(int)


for col in [
    'kategorie_num_mom',    
]:
    rbs_sample = make_dichot(rbs_sample, col)

rbs_sample.to_pickle(target1)
rbs_sample.to_excel(target2, index=False)


