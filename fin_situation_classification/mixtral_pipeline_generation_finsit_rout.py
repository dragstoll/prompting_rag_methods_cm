# Mixtral Generation
# !pip install git+https://github.com/huggingface/transformers -q peft  accelerate bitsandbytes safetensors sentencepiece
# !pip install -U pandas

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# BitsAndBytesConfig
import pandas as pd

import argparse, shutil, logging
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,)
parser.add_argument('--target1', type=str,)
parser.add_argument('--target2', type=str,)

args = parser.parse_args()
source = args.source 
target1 = args.target1
target2 = args.target2
# print(source)
# print(target1)
# print(target2)
import sys
# sys.exit()

# rbs_alk_eltern_sample = pd.read_pickle('rbs/rbs0-4000.pkl')
# rbs_alk_eltern_sample = pd.read_pickle(source)
rbs_alk_eltern_sample = pd.read_pickle(source)
# rbs_alk_eltern_sample = pd.read_pickle('rbs1k/rbs7000-8000.pkl')
# rbs_alk_eltern_sample = rbs_alk_eltern_sample.sample(2).reset_index(drop=True).copy()
# print(rbs_alk_eltern_sample.columns)
import pandas as pd
from transformers import AutoTokenizer
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"



# tokenizer = AutoTokenizer.from_pretrained(model_name)

# rbs_alk_eltern_sample['tokennum'] = rbs_alk_eltern_sample['text'].apply(lambda x: len(tokenizer.tokenize(x)))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

# rbs_alk_eltern_sample['text_shot'] = rbs_alk_eltern_sample['text'].apply(lambda x: x[:7000] if len(tokenizer.tokenize(x)) > 7000 else x)

# rbs_alk_eltern_sample['tokennum_shot'] = rbs_alk_eltern_sample['text_shot'].apply(lambda x: len(tokenizer.tokenize(x)))

# rbs_alk_eltern_sample['iteration'] = range(1, len(rbs_alk_eltern_sample) + 1)
# max_value = rbs_alk_eltern_sample['tokennum_shot'].max()
# print(max_value)




# Define stop token ids
stop_token_ids = [0]

from transformers import LlamaTokenizerFast, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your chat template
chat_template = [{"role": "system", "content": 'You are chatting with an AI assistant.'}, {"role": "user", "content": "{user_input}"}]

# Set the chat template for the tokenizer
tokenizer.chat_template = chat_template



# text2 = rbs_alk_eltern_sample[rbs_alk_eltern_sample['psych_begriff_vorhanden'] == 1]['text'].values[0]
# print(text2)

# text0 = "[INST] Gibt es Hinweise für Alkohol Probleme bei den Eltern in diesem Rechenschaftsbericht (Alkoholkrankheit, Alkoholsucht, Alkoholabhängigkeit, Alkoholentzug, Alkohol Therapie, Alkoholtests, Alkoholkontrollen) aktuell oder in der Vergangenheit, auch wenn die Probleme überwunden sind? \nRechenschaftsbericht:\n"
# text3= " [/INST]"

prompt_template0 = """
<s>[INST] 
Instruction: Answer the question based on provided context, do not interpret or speculate. Here is context to help:

"""
# {context}
prompt_template1 = """
### QUESTION:
Gibt es Hinweise für schwierige finanzielle Situation in diesem Rechenschaftsbericht? \nA: Hinweise vorhanden für schwierige finanzielle Situation \nB: Hinweise nicht vorhanden für schwierige finanzielle Situation  \nListe klare, explizite Hinweise für die Einschätzung.

[/INST]
"""
# alte Version vom prompt 
# Wie ist die Entwicklung, Situation des Kindes bei der Beistandschaft in diesem Rechenschaftsbericht, gemäss dem Auftrag, der Beurteilung, Prognose, den Schlussfolgerungen, Zielen und Anträgen? \nA: sich verbessernde Entwicklung \nB: gleichbleibende, sich nicht verbessernde Entwicklung  \nC: sich verschlechternde Entwicklung \nListe klare, explizite Hinweise für die Einschätzung.
#  \nListe klare, explizite Hinweise für die Einschätzung.

# text= text0+text2+text3
# print(text)
rbs_alk_eltern_sample['prompt'] = rbs_alk_eltern_sample['text'].apply(lambda x: prompt_template0 + x + prompt_template1)

from transformers import TextGenerationPipeline




bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16
)
   
pipe = transformers.pipeline(
  "text-generation",
  model=model_name,
  model_kwargs={"torch_dtype": torch.float16, "quantization_config": bnb_config, "device_map": "auto", "attn_implementation": "flash_attention_2",},
  max_new_tokens=300, do_sample=True, temperature=0.1, repetition_penalty=1.1, top_k=50, top_p=0.95,  pad_token_id=tokenizer.eos_token_id, return_full_text=False, 
#   tokenizer_kwargs={"pad_token_id": "eos_token_id"}
  # device="cuda:0",
  
)
# pipe.tokenizer.pad_token_id = model.config.eos_token_id


from datasets import Dataset
#sample 30 rows
# rbs_alk_eltern_sample=rbs_alk_eltern_sample.sample(30).reset_index(drop=True).copy()

dataset = Dataset.from_pandas(rbs_alk_eltern_sample)



from transformers.pipelines.pt_utils import KeyDataset

out_list = []
for out in pipe(KeyDataset(dataset, "prompt"), ):    
    # print(out)
    out_list.append(out)
    # dataset_output = dataset.map(lambda examples: {'output': out})
    

# rbs_alk_eltern_sample10["antwort_dict"]=out_list.copy()
# rbs_alk_eltern_sample10["antwort"]=rbs_alk_eltern_sample10["antwort_dict"].apply(lambda x: x[0]['generated_text'])
# rbs_alk_eltern_sample10["antwort"]

# # Save rbs_alk_eltern_sample10 as an Excel file
# rbs_alk_eltern_sample10.to_excel('rbs_alk_eltern_sample10.xlsx', index=False)

rbs_alk_eltern_sample["antwort_dict"]=out_list.copy()
rbs_alk_eltern_sample["antwort"]=rbs_alk_eltern_sample["antwort_dict"].apply(lambda x: x[0]['generated_text'])
rbs_alk_eltern_sample["antwort"]

# Save rbs_alk_eltern_sample10 as an Excel file
# rbs_alk_eltern_sample.to_excel('rbs_alk_eltern_sample_antwort.xlsx', index=False)

import sys
# sys.exit()

# dataset = Dataset.from_pandas(rbs_alk_eltern_sample10)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name ="deepset/gelectra-large"
# Replace this with your own checkpoint
tokenizer = AutoTokenizer.from_pretrained("finsit_class_gelectra_direkt", do_lower_case = True,
                                         use_fast=True, max_length=512, truncation=True, padding=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("finsit_class_gelectra_direkt").to("cuda")

# predict_class = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

# device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# text=str(rbs_alk_eltern_sample10["antwort"][0:1])
# inputs = tokenizer(text, return_tensors="pt")
# inputs=inputs.to(device)
# print(inputs)


# with torch.no_grad():
#     logits = model(**inputs)


# label= torch.argmax(logits[0]).item()
# print(label)
rbs_alk_eltern_sample['label'] = None

rbs_alk_eltern_sample.reset_index(inplace=True)

model.eval()
for i in range(len(rbs_alk_eltern_sample)): 
         
        text=str(rbs_alk_eltern_sample.loc[i, "antwort"])
        # print(text)
        # inputs = tokenizer(text, return_tensors="pt")
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs=inputs.to(device)
        # print(inputs)
        with torch.no_grad():
            logits = model(**inputs)
        predicted_class_id = torch.argmax(logits[0]).item()
        # print(predicted_class_id)
        rbs_alk_eltern_sample.loc[i, "label"]=predicted_class_id
        
        

pd.options.display.max_colwidth = 1000
# print(rbs_alk_eltern_sample[["antwort", "label"]])

rbs_alk_eltern_sample_codiert=rbs_alk_eltern_sample.copy()


# Save DataFrame as pkl file
# rbs_alk_eltern_sample_codiert.to_pickle('rbs/rbs0-4000_psych_codiert.pkl')
rbs_alk_eltern_sample_codiert.to_pickle(target1)

# Save DataFrame as Excel file
# rbs_alk_eltern_sample_codiert.to_excel('rbs/rbs0-4000_psych_codiert.xlsx', index=False)
rbs_alk_eltern_sample_codiert.to_excel(target2, index=False)
