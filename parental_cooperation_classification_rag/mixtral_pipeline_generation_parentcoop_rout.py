# Mixtral Generation
# !pip install git+https://github.com/huggingface/transformers -q peft  accelerate bitsandbytes safetensors sentencepiece
# !pip install -U pandas

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import pandas as pd
import argparse, shutil, logging

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,)
parser.add_argument('--target1', type=str,)
parser.add_argument('--target2', type=str,)
args = parser.parse_args()
source = args.source 
target1 = args.target1
target2 = args.target2

# Load the dataset from the source file
rbs_alk_eltern_sample = pd.read_pickle(source)

# Initialize the tokenizer
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt templates
prompt_template0 = """
<s>[INST] 
Instruction: Answer the question based on provided context, do not interpret or speculate. Here is context to help:

"""
prompt_template1 = """
### QUESTION:
Gibt es Hinweise für mangelnde Bereitschaft, Widerstand oder Verweigerung der Zusammenarbeit oder Kontaktverweigerung seitens der Eltern gegenüber der Massnahme, der Beistandschaft, den Beiständen, den Anweisungen in diesem Rechenschaftsbericht? \nA: Hinweise vorhanden für mangelnde Bereitschaft, Widerstand oder Verweigerung der Zusammenarbeit oder Kontaktverweigerung gegenüber der Massnahme, der Beistandschaft, den Beiständen, den Anweisungen \nB: Hinweise nicht vorhanden für mangelnde Bereitschaft, Widerstand oder Verweigerung der Zusammenarbeit oder Kontaktverweigerung gegenüber der Massnahme, der Beistandschaft, den Beiständen, den Anweisungen  \nListe klare, explizite Hinweise für die Einschätzung.

[/INST]
"""

# Generate prompts for each text in the dataset
rbs_alk_eltern_sample['prompt'] = rbs_alk_eltern_sample['text'].apply(lambda x: prompt_template0 + x + prompt_template1)

# Configure BitsAndBytes for model quantization
bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16
)

# Initialize the text generation pipeline
pipe = transformers.pipeline(
  "text-generation",
  model=model_name,
  model_kwargs={"torch_dtype": torch.float16, "quantization_config": bnb_config, "device_map": "auto", "attn_implementation": "flash_attention_2",},
  max_new_tokens=300, do_sample=True, temperature=0.1, repetition_penalty=1.1, top_k=50, top_p=0.95,  pad_token_id=tokenizer.eos_token_id, return_full_text=False, 
)

# Convert the dataset to Hugging Face Dataset format
from datasets import Dataset
dataset = Dataset.from_pandas(rbs_alk_eltern_sample)

# Generate text for each prompt in the dataset
from transformers.pipelines.pt_utils import KeyDataset
out_list = []
for out in pipe(KeyDataset(dataset, "prompt")):    
    out_list.append(out)

# Add generated text to the dataset
rbs_alk_eltern_sample["antwort_dict"] = out_list.copy()
rbs_alk_eltern_sample["antwort"] = rbs_alk_eltern_sample["antwort_dict"].apply(lambda x: x[0]['generated_text'])

# Initialize the classification model and tokenizer
model_name = "deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained("parentcoop_class_gelectra_direkt", do_lower_case=True, use_fast=True, max_length=512, truncation=True, padding=True, eos_token='###', pad_token='[PAD]',)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained("parentcoop_class_gelectra_direkt").to("cuda")

# Set device for computation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Initialize label column
rbs_alk_eltern_sample['label'] = None
rbs_alk_eltern_sample.reset_index(inplace=True)

# Perform classification for each generated text
model.eval()
for i in range(len(rbs_alk_eltern_sample)): 
    text = str(rbs_alk_eltern_sample.loc[i, "antwort"])
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs)
    predicted_class_id = torch.argmax(logits[0]).item()
    rbs_alk_eltern_sample.loc[i, "label"] = predicted_class_id

# Save the processed dataset to files
rbs_alk_eltern_sample_codiert = rbs_alk_eltern_sample.copy()
rbs_alk_eltern_sample_codiert.to_pickle(target1)
rbs_alk_eltern_sample_codiert.to_excel(target2, index=False)
