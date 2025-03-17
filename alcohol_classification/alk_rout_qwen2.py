# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import pandas as pd
import os

# Set environment variables for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import additional libraries
import argparse, shutil, logging

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,)
parser.add_argument('--target1', type=str,)
parser.add_argument('--target2', type=str,)
parser.add_argument('--iter', type=str,)

# Parse arguments
args = parser.parse_args()
source = args.source 
target1 = args.target1
target2 = args.target2
iter = args.iter

# Load the source data
sample_df = pd.read_pickle(source)

# Import additional libraries
import sys
import pandas as pd
from transformers import AutoTokenizer

# Set model name
model_name = "unsloth/Qwen2-72B-Instruct-bnb-4bit"

# Load model configuration
model_config = transformers.AutoConfig.from_pretrained(model_name)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set up bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Set up quantization config
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

# Set model and encoding kwargs
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': False}

# Load embedding model
embedmodel_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct'
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embedhf= HuggingFaceEmbeddings(model_name=embedmodel_name, encode_kwargs=encode_kwargs)

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(
   model_name,
   quantization_config=bnb_config,
   attn_implementation="flash_attention_2",
   device_map="auto",
)

# Import additional libraries
import os
import pandas as pd
import random

# Import langchain and other necessary libraries
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
import nest_asyncio
nest_asyncio.apply()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader

# Set up text generation pipeline
text_generation_pipeline = transformers.pipeline(
   model=model,
   tokenizer=tokenizer,
   task="text-generation",
   temperature=0.1,
   repetition_penalty=1.1,
   max_new_tokens=300,
   do_sample=True,
   top_k=50, top_p=0.90, 
   pad_token_id=tokenizer.eos_token_id, 
   return_full_text=False,
)

# Define prompt template
prompt_template = """
<s> [INST] 
Anweisung: Interpretiere nicht, spekuliere nicht, was sein könnte, sondern du musst die Frage anhand des vorgegebenen Kontexts beantworten.
## KONTEXT:
{context}

{question} [/INST]
"""

# Create HuggingFace pipeline
mixtral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
   input_variables=["context", "question"],
   template=prompt_template,
)

# Create LLM chain
llm_chain = LLMChain(llm=mixtral_llm, prompt=prompt)

# Import additional libraries
from langchain_core.runnables import RunnablePassthrough

# Define query and definition
query = """
## FRAGE:
Gibt es explizit erwähnte, ausdrückliche Hinweise für Alkohol Probleme bei den Eltern oder den Partnern, Partnerinnen der Eltern in diesem Rechenschaftsbericht 
(Alkoholkrankheit, Alkoholsucht, Alkoholabhängigkeit, Alkoholentzug, Alkohol Therapie, Alkoholtests, Alkoholkontrollen) aktuell oder in der Vergangenheit, auch wenn die Probleme überwunden sind?

Wenn ja, liste die Hinweise auf.
"""

definition= """
Gibt es explizit erwähnte, ausdrückliche Hinweise für Alkohol Probleme bei den Eltern oder den Partnern, Partnerinnen der Eltern in diesem Rechenschaftsbericht (Alkoholkrankheit, Alkoholsucht, Alkoholabhängigkeit, Alkoholentzug, Alkohol Therapie, Alkoholtests, Alkoholkontrollen) aktuell oder in der Vergangenheit, auch wenn die Probleme überwunden sind?
"""

# Set logging level to ERROR
import logging
logging.getLogger().setLevel(logging.ERROR)

# Import additional libraries
import gc
from llama_index.core.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.core.utils import truncate_text

# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

# Function to display source node
def display_source_node(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> None:
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        f"Node ID: {source_node.node.node_id} \n"
        f"Score: {source_node.score} \n"
        f"Text: {source_text_fmt} \n"
    )
    if show_source_metadata:
        text_md += f"Metadata: {source_node.node.metadata} \n"
    if isinstance(source_node.node, ImageNode):
        text_md += "Image:"

    print(text_md)

# Import additional libraries
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import QueryBundle
import spacy
from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Define Document2 class
class Document2:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = ''

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

# Function to get answer
def get_answer(text):
    with open(iter+'rb.txt', 'w') as file:
        file.write(text)

    loader = TextLoader(iter+"rb.txt")
    docs_transformed = loader.load()    
    chunked_documents = text_splitter.split_documents(docs_transformed)

    db = FAISS.from_documents(chunked_documents, embedhf)
    retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 10, },    
    )
    docs = retriever.get_relevant_documents(definition)
    list_chunks = [doc.page_content for doc in docs]
    list_chunks2 = [] 

    try:
        rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
        )
    
        result= rag_chain.invoke(query)
        torch.cuda.empty_cache()
        gc.collect()    
        return result['text'], result['context'], list_chunks, list_chunks2
    except IndexError:
        return "Error", list_chunks, list_chunks, list_chunks2

# Function to extract context
def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("',")[0] for doc in text.split("Document(")[1:]]
        extracted_context = '\n'.join(extracted_texts)
        extracted_context = '\nChunk: '.join(line for line in extracted_context.splitlines() if line.strip())
        return extracted_context
    except IndexError:
        return None

# Set logging level to ERROR
import logging
logging.getLogger().setLevel(logging.ERROR)

# Initialize variables
context = {}
answer = []
list_chunks1 = []
list_chunks2 = []

# Iterate through sample_df and get answers
for index, row in sample_df.iterrows():
    text = row['text']
    antwort, kontext, list1, list2 = get_answer(text)
    answer.append(antwort)
    context[index] = kontext
    list_chunks1.append(list1)
    list_chunks2.append(list2)

# Add answers and context to sample_df
sample_df['antwort'] = answer
sample_df['context'] = context
sample_df['list_chunks1'] = list_chunks1
sample_df['list_chunks2'] = list_chunks2

# Save sample_df to Excel
sample_df.to_excel(iter+'sample_dfTemp2.xlsx')
sample_df0= pd.read_excel(iter+'sample_dfTemp2.xlsx', sheet_name='Sheet1')
sample_df0['extracted_context'] = sample_df0['context'].apply(lambda x: extract_context(x))
sample_df=sample_df0.copy()

# Save final DataFrame to Excel and pickle
rbs_alk_eltern_sample=sample_df.copy()
file_path = iter+'sample_dfTemp2.xlsx'
if os.path.exists(file_path):
  os.remove(file_path)

# Import additional libraries
import sys

# Import necessary libraries for model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast, AutoTokenizer
import os

# Set model name and load tokenizer
model_name ="deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained("alk_class", do_lower_case = True,
                                         use_fast=True, max_length=512, truncation=True, padding=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForSequenceClassification.from_pretrained("alk_class").to("cuda")

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize label column
rbs_alk_eltern_sample['label'] = None
rbs_alk_eltern_sample.reset_index(inplace=True)

# Evaluate model and predict labels
model.eval()
for i in range(len(rbs_alk_eltern_sample)): 
        text=str(rbs_alk_eltern_sample.loc[i, "antwort"])
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        inputs=inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs)
        predicted_class_id = torch.argmax(logits[0]).item()
        rbs_alk_eltern_sample.loc[i, "label"]=predicted_class_id

# Save final DataFrame to pickle and Excel
rbs_alk_eltern_sample_codiert=rbs_alk_eltern_sample.copy()
rbs_alk_eltern_sample_codiert.to_pickle(target1)
rbs_alk_eltern_sample_codiert.to_excel(target2, index=False)
