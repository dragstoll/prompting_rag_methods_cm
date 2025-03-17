# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# BitsAndBytesConfig is used for model quantization
import pandas as pd
import os

# Enable dynamic memory allocation for CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up command line arguments
import argparse, shutil, logging
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str,)  # Source data file path
parser.add_argument('--target1', type=str,)  # Target pickle file path
parser.add_argument('--target2', type=str,)  # Target Excel file path
parser.add_argument('--iter', type=str,)  # Iteration identifier for intermediate files

# Parse command line arguments
args = parser.parse_args()
source = args.source 
target1 = args.target1
target2 = args.target2
iter = args.iter

# Debug prints
# print(source)
# print(target1)
# print(target2)
import sys

# Load the dataset from the source file
sample_df = pd.read_pickle(source)
# Optional: Sample a subset of the data for testing
# sample_df = sample_df.sample(20).reset_index(drop=True).copy()

# Import AutoTokenizer again (redundant)
import pandas as pd
from transformers import AutoTokenizer

# Define LLM model to use
# model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
model_name = "unsloth/Qwen2-72B-Instruct-bnb-4bit"  # Using Qwen2-72B model optimized with 4-bit quantization
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load model configuration
model_config = transformers.AutoConfig.from_pretrained(
   model_name,
)

# Initialize tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
tokenizer.padding_side = "right"  # Pad on the right side

#################################################################
# BitsAndBytes quantization parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models (float16 for better performance)
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"  # Normal float 4-bit quantization

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
# Convert string dtype to torch dtype
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Create BitsAndBytes configuration for model loading
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
       
# Embedding model configuration options (commented out)
# model_kwargs = {'device': 'cuda:0'}
# model_kwargs = { 'quantization_config': bnb_config, 'attn_implementation': 'flash_attention_2', 'device_map': 'auto' }
                
# Embedding settings
encode_kwargs = {'normalize_embeddings': False}

# Define embedding model name
# embedmodel_name='intfloat/multilingual-e5-large'
embedmodel_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct'

# Import embedding related libraries
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
# Initialize embedding model
embedhf= HuggingFaceEmbeddings(
    model_name=embedmodel_name,
    encode_kwargs=encode_kwargs
)

#################################################################
# Load pre-trained LLM
#################################################################
# Load the large language model with optimizations
model = AutoModelForCausalLM.from_pretrained(
   model_name,
   quantization_config=bnb_config,
   attn_implementation="flash_attention_2",  # Use Flash Attention 2 for faster inference
   device_map="auto",  # Automatically distribute model across available GPUs
)

# Import additional libraries for document processing and RAG
import os
import pandas as pd
import random
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

# Create a text generation pipeline with the loaded model
text_generation_pipeline = transformers.pipeline(
   model=model,
   tokenizer=tokenizer,
   task="text-generation",
   temperature=0.1,  # Low temperature for more deterministic outputs
   repetition_penalty=1.1,  # Penalize repetition slightly
   max_new_tokens=512,  # Maximum number of tokens to generate
   do_sample=True,  # Use sampling
   top_k=50, top_p=0.90,  # Top-k and nucleus sampling parameters
   pad_token_id=tokenizer.eos_token_id, 
   return_full_text=False,  # Only return the newly generated text
)

# Define the prompt template in German
prompt_template = """
<s> [INST] 
Anweisung: 
Interpretiere nicht, spekuliere nicht, was sein könnte, sondern du musst die Frage anhand des vorgegebenen Kontexts beantworten.
Zuerst musst du die Frage mit A, B oder C beantworten. 
Dann musst du klare, explizite Hinweise für die Einschätzung auflisten.

## KONTEXT:
{context}


{question} [/INST]
"""

# Initialize the LLM with the pipeline
mixtral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
   input_variables=["context", "question"],
   template=prompt_template,
)

# Create LLM chain
llm_chain = LLMChain(llm=mixtral_llm, prompt=prompt)

# Import RunnablePassthrough for RAG pipeline
from langchain_core.runnables import RunnablePassthrough

# Define the query to be asked for each document
query = """
## FRAGE:
Wie entwickelt sich Situation des Kindes bezüglich der im Text beschriebenen Ausgangslage, dem Auftrag, den Zielen der Beistandschaft, gemäss der Berichterstattung, der Beurteilung, der Prognose, den Schlussfolgerungen, den Anträgen in diesem Rechenschaftsbericht? 

A: sich verbessernde Entwicklung 
B: gleichbleibende, sich nicht verbessernde Entwicklung oder uneinheitliche, unklare Entwicklung  
C: sich verschlechternde Entwicklung 

"""

# Define the core question for document retrieval
definition= """
Wie entwickelt sich Situation des Kindes bezüglich der im Text beschriebenen Ausgangslage, dem Auftrag, den Zielen der Beistandschaft, gemäss der Berichterstattung, der Beurteilung, der Prognose, den Schlussfolgerungen, den Anträgen in diesem Rechenschaftsbericht? 
"""

# Set logging level to ERROR to reduce output noise
import logging
logging.getLogger().setLevel(logging.ERROR)

# Import garbage collection for memory management
import gc

# Import LlamaIndex utilities for document processing
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

# Function to display source nodes with metadata
def display_source_node(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> None:
    """Display source node"""
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

# Import reranker and other text splitting utilities
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import QueryBundle
import spacy
from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Custom Document class for handling text chunks
class Document2:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = ''

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"

# Main function to process text and get answer using RAG
def get_answer(text):
    # Save the text to a temporary file
    with open(iter+'rb.txt', 'w') as file:
        file.write(text)

    # Load the text using TextLoader
    loader = TextLoader(iter+"rb.txt")
    docs_transformed = loader.load()    
    
    # Split documents into chunks
    chunked_documents = text_splitter.split_documents(docs_transformed)
   
    # Create a FAISS vector store from the chunks
    db = FAISS.from_documents(chunked_documents, embedhf)
    
    # Set up retriever for semantic search
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 10, },    
    )
    
    # Retrieve relevant documents based on the query
    docs = retriever.get_relevant_documents(definition)
    
    # Extract content from retrieved documents
    list_chunks = [doc.page_content for doc in docs]
    list_chunks2 = [] 
    
    # Set up and run the RAG chain
    try:
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | llm_chain
        )
    
        # Get result from the RAG chain
        result = rag_chain.invoke(query)
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()    
        
        return result['text'], result['context'], list_chunks, list_chunks2
    except IndexError:
        return "Error", list_chunks, list_chunks, list_chunks2

# Function to extract context from document output
def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("',")[0] for doc in text.split("Document(")[1:]]
        extracted_context = '\n'.join(extracted_texts)
        extracted_context = '\nChunk: '.join(line for line in extracted_context.splitlines() if line.strip())
        return extracted_context
    except IndexError:
        return None
    
# Set logging level to ERROR to reduce output noise
import logging
logging.getLogger().setLevel(logging.ERROR)

# Initialize containers for results
context = {}
answer = []
list_chunks1 = []
list_chunks2 = []

# Process each document in the dataset
for index, row in sample_df.iterrows():
    text = row['text']
    antwort, kontext, list1, list2 = get_answer(text)
    answer.append(antwort)
    context[index] = kontext
    list_chunks1.append(list1)
    list_chunks2.append(list2)
    
# Add results to the dataframe
sample_df['antwort'] = answer
sample_df['context'] = context
sample_df['list_chunks1'] = list_chunks1
sample_df['list_chunks2'] = list_chunks2

# Save intermediate results to Excel
sample_df.to_excel(iter+'sample_dfTemp2.xlsx')
sample_df0 = pd.read_excel(iter+'sample_dfTemp2.xlsx', sheet_name='Sheet1')

# Extract and process context
sample_df0['extracted_context'] = sample_df0['context'].apply(lambda x: extract_context(x))
sample_df = sample_df0.copy()

# Create a copy for classification
rbs_alk_eltern_sample = sample_df.copy()

# Clean up temporary files
file_path = iter+'sample_dfTemp2.xlsx'
if os.path.exists(file_path):
  os.remove(file_path)

# Import libraries for classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast, AutoTokenizer
import os

# Define classification model
model_name = "deepset/gelectra-large"

# Load pre-trained tokenizer for classification
tokenizer = AutoTokenizer.from_pretrained("fallentwicklung_class_qwen2", do_lower_case=True,
                                         use_fast=True, max_length=512, truncation=True, padding=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
# Add special tokens to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

# Load pre-trained classification model
model = AutoModelForSequenceClassification.from_pretrained("fallentwicklung_class_qwen2").to("cuda")

# Set device for inference
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize label column
rbs_alk_eltern_sample['label'] = None
rbs_alk_eltern_sample.reset_index(inplace=True)

# Run classification on each document
model.eval()
for i in range(len(rbs_alk_eltern_sample)): 
    # Get text from answer column    
    text = str(rbs_alk_eltern_sample.loc[i, "antwort"])
    
    # Tokenize and prepare for model
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = inputs.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(**inputs)
    predicted_class_id = torch.argmax(logits[0]).item()
    
    # Save prediction to dataframe
    rbs_alk_eltern_sample.loc[i, "label"] = predicted_class_id

# Set display options
pd.options.display.max_colwidth = 1000

# Create a copy for final output
rbs_alk_eltern_sample_codiert = rbs_alk_eltern_sample.copy()

# Save results to pickle and Excel files
rbs_alk_eltern_sample_codiert.to_pickle(target1)
rbs_alk_eltern_sample_codiert.to_excel(target2, index=False)
