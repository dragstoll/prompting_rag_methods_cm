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

# rbs_psych_eltern_sample = pd.read_pickle('rbs/rbs0-4000.pkl')
rbs_psych_eltern_sample = pd.read_pickle(source)
# rbs_psych_eltern_sample=rbs_psych_eltern_sample.sample(10)
# print(rbs_psych_eltern_sample.columns)
import pandas as pd
from transformers import AutoTokenizer
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"



model_config = transformers.AutoConfig.from_pretrained(
   model_name,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
   load_in_4bit=use_4bit,
   bnb_4bit_quant_type=bnb_4bit_quant_type,
   bnb_4bit_compute_dtype=compute_dtype,
   bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
   major, _ = torch.cuda.get_device_capability()
   if major >= 8:
       print("=" * 80)
       print("Your GPU supports bfloat16: accelerate training with bf16=True")
       print("=" * 80)
       

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
   model_name,
   quantization_config=bnb_config,
   attn_implementation="flash_attention_2",
)


import os


import pandas as pd

import random
# sample_df = pd.read_pickle('traintest_dataset_rbs_psych.pkl')

from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import nest_asyncio
nest_asyncio.apply()
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader

text_generation_pipeline = transformers.pipeline(
   model=model,
   tokenizer=tokenizer,
   task="text-generation",
   temperature=0.2,
   repetition_penalty=1.1,
#    return_full_text=True,
   max_new_tokens=300,
   do_sample=True,
   top_k=50, top_p=0.90, 
   pad_token_id=tokenizer.eos_token_id, return_full_text=False,
)

prompt_template = """
<s>[INST] 
Instruction: Answer the question based on provided context, do not interpret or speculate. Here is context to help:

{context}

### QUESTION:
{question}

[/INST]
"""

mixtral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
   input_variables=["context", "question"],
   template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mixtral_llm, prompt=prompt)


# llm_chain.invoke({"context":"",
#                  "question": "What were the two main things the author worked on before college?"})

from langchain_core.runnables import RunnablePassthrough
query = "Liste klare, explizite Hinweise für psychische Störungen, psychische Erkrankungen, Persönlichkeitsstörungen, psychologische oder psychiatrische Behandungen bei den Eltern oder deren Partner, auch in der Vergangenheit? Damit ist gemeint: psychologische oder psychiatrische Therapie/Betreuung, welche besucht, verweigert wird oder empfohlen wird, Einweisung in eine psychiatrische Klinik, wenn psychische, psychiatrische Diagnosen diagnostiziert sind wie: Depressionen, Anststörungen, Wahn, Bipolar, Borderline Persönlichkeitsstörungen."



import logging
logging.getLogger().setLevel(logging.ERROR)


# answer = []
# for index, row in sample_20.iterrows():
#     text = row['text']
#     antwort = get_answer(text)
#     answer.append(antwort)
    

# sample_20['antwort'] = answer

# # sample_20.to_pickle('sample_df.pkl')
# sample_20.to_excel('sample_df_llammaIndex.xlsx')

import spacy
from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#  text_splitter = SpacyTextSplitter(pipeline="de_core_news_lg", chunk_size=100, chunk_overlap=5)

def get_answer(text):
    with open('rb.txt', 'w') as file:
        file.write(text)

    loader = TextLoader("rb.txt")
    # loader = TextLoader(text)
    docs_transformed = loader.load()    
    chunked_documents = text_splitter.split_documents(docs_transformed)
   #  chunked_documents = text_splitter.create_documents(docs_transformed)
    
    db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large'))
    retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'fetch_k': 20, 'k': 10}
    )


    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
    )
    result= rag_chain.invoke(query)
    # print(result['text'])
    return result['text'], result['context']


def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("',")[0] for doc in text.split("Document(")[1:]]
        # Print the extracted texts (you can save them or process further as needed)
        # print(extracted_texts)
        extracted_context = '\n'.join(extracted_texts)
        extracted_context = '\nChunk: '.join(line for line in extracted_context.splitlines() if line.strip())
      #   print(extracted_context)
        return extracted_context
    except IndexError:
        # print("IndexError: list index out of range")
        return None
    
import logging
logging.getLogger().setLevel(logging.ERROR)

context = {}
answer = []
for index, row in rbs_psych_eltern_sample.iterrows():
    text = row['rbs_aggregated']
    antwort, kontext = get_answer(text)
    answer.append(antwort)
    context[index] = kontext
    

rbs_psych_eltern_sample['antwort'] = answer
rbs_psych_eltern_sample['context'] = context


rbs_psych_eltern_sample.to_pickle('rbs_psych_eltern_sampleTemp.pkl')
rbs_psych_eltern_sample.to_excel('rbs_psych_eltern_sampleTemp.xlsx')
rbs_psych_eltern_sample0= pd.read_excel('rbs_psych_eltern_sampleTemp.xlsx', sheet_name='Sheet1')
rbs_psych_eltern_sample0['extracted_context'] = rbs_psych_eltern_sample0['context'].apply(lambda x: extract_context(x))
rbs_psych_eltern_sample0.to_excel('rbs_psych_eltern_sampleTemp.xlsx')
rbs_psych_eltern_sample=rbs_psych_eltern_sample0.copy()

# query = "Liste klare, explizite Hinweise für psychische Störungen, psychische Erkrankungen, Persönlichkeitsstörungen, psychologische oder psychiatrische Behandungen bei den Eltern oder deren Partner, auch in der Vergangenheit? Damit ist gemeint: psychologische oder psychiatrische Therapie/Betreuung, welche besucht, verweigert wird oder empfohlen wird, Einweisung in eine psychiatrische Klinik, wenn psychische, psychiatrische Diagnosen diagnostiziert sind wie: Depressionen, Anststörungen, Wahn, Bipolar, Borderline Persönlichkeitsstörungen."
# rbs_psych_eltern_sample['antwort']=query + "\nAntwort:\n" + rbs_psych_eltern_sample['antwort']
# print(rbs_psych_eltern_sample['antwort'][0:1])
# Save rbs_psych_eltern_sample10 as an Excel file
# rbs_psych_eltern_sample.to_excel('rbs_psych_eltern_sample_antwort.xlsx', index=False)

import sys
# sys.exit()

# dataset = Dataset.from_pandas(rbs_psych_eltern_sample10)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name ="deepset/gelectra-large"
# Replace this with your own checkpoint
tokenizer = AutoTokenizer.from_pretrained("psych_class_gelectra_rag_recursFaiss2", do_lower_case = True,
                                         use_fast=True,
                                         eos_token='###', pad_token='[PAD]', max_length=512, padding="max_length")
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("psych_class_gelectra_rag_recursFaiss2").to("cuda")

predict_class = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

# device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(device)

# text=str(rbs_psych_eltern_sample10["antwort"][0:1])
# inputs = tokenizer(text, return_tensors="pt")
# inputs=inputs.to(device)
# print(inputs)


# with torch.no_grad():
#     logits = model(**inputs)


# label= torch.argmax(logits[0]).item()
# print(label)
rbs_psych_eltern_sample['label'] = None

rbs_psych_eltern_sample.reset_index(inplace=True)

model.eval()
for i in range(len(rbs_psych_eltern_sample)): 
         
        text=str(rbs_psych_eltern_sample.loc[i, "antwort"])
        # print(text)
        # inputs = tokenizer(text, return_tensors="pt", max_length=512, padding="max_length")
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs=inputs.to(device)
        # print(inputs)
        with torch.no_grad():
            logits = model(**inputs)
        predicted_class_id = torch.argmax(logits[0]).item()
        # print(predicted_class_id)
        rbs_psych_eltern_sample.loc[i, "label"]=predicted_class_id
        
        

pd.options.display.max_colwidth = 1000
# print(rbs_psych_eltern_sample[["antwort", "label"]])

rbs_psych_eltern_sample_codiert=rbs_psych_eltern_sample.copy()


# Save DataFrame as pkl file
# rbs_psych_eltern_sample_codiert.to_pickle('rbs/rbs0-4000_psych_codiert.pkl')
rbs_psych_eltern_sample_codiert.to_pickle(target1)

# Save DataFrame as Excel file
# rbs_psych_eltern_sample_codiert.to_excel('rbs/rbs0-4000_psych_codiert.xlsx', index=False)
rbs_psych_eltern_sample_codiert.to_excel(target2, index=False)
