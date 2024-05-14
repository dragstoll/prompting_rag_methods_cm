
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
# BitsAndBytesConfig
import pandas as pd
import os

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

# print(source)
# print(target1)
# print(target2)
import sys

sample_df = pd.read_pickle(source)
# sample_df = sample_df.sample(2).reset_index(drop=True).copy()

# sample_df2 = pd.read_excel('rbs_ecan_klient_sample1-500.xlsx', sheet_name='Sheet1')
# sample_df = pd.read_excel('rbs_ecan_klient_sample500-1000.xlsx', sheet_name='Sheet1')
# sample_df = pd.concat([sample_df2, sample_df])

# sample_df = sample_df[sample_df['ecan_begriff_vorhanden'] == 1].sample(10).reset_index(drop=True).copy()

# sample_df.columns   

# rbs_alk_eltern_sample=pd.read_pickle(iter+'rbs_psych_eltern_sampleTemp.pkl')

# file_path = iter + "rbs_psych_eltern_sampleTemp.pkl"
# if os.path.exists(file_path):
#   os.remove(file_path)

# rbs_alk_eltern_sample= pd.read_excel('rbs_ecan_klient_sample_antwort_recursFaissRerankBGE500-1000.xlsx', sheet_name='Sheet1')

import pandas as pd
from transformers import AutoTokenizer
model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'
# model_name='mistral-community/Mixtral-8x22B-v0.1-4bit'

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
   device_map="auto",
)
import pandas as pd
import random
import random


from langchain_community.llms import HuggingFacePipeline


text_generation_pipeline = transformers.pipeline(
   model=model,
   tokenizer=tokenizer,
   task="text-generation",
   temperature=0.1,
   repetition_penalty=1.1,
#    return_full_text=True,
   max_new_tokens=512,
   do_sample=True,
   top_k=50, top_p=0.90, 
   pad_token_id=tokenizer.eos_token_id, return_full_text=False,
)


mixtral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)



from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import  ServiceContext
# from llama_index import LangchainEmbedding, ServiceContext

embed_model =   HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from transformers import AutoTokenizer
from llama_index.core import set_global_tokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1").encode  # pass in the HuggingFace model org + repo
)

# from llama_index.core import ServiceContext

service_context = ServiceContext.from_defaults(
    llm=mixtral_llm,
    embed_model=embed_model,
    system_prompt="""
<s>[INST]
### ANWEISUNG:  
Beantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.

### KONTEXT: 

"""    
)

from pathlib import Path
import glob
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex






from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)


rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=10)


from llama_index.core.node_parser import (SentenceSplitter, SemanticSplitterNodeParser,)

# splitter = SemanticSplitterNodeParser(
#     buffer_size=1, breakpoint_percentile_threshold=95, embed_model=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
# )
text_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=0.1,)


def get_answer(text):
    with open(iter+'rb.txt', 'w') as file:
        file.write(text)
    rbname=iter+'rb.txt'
    transcript_files = glob.glob(str(Path("./") / rbname), recursive=True)
    
    
    # transcript_files = glob.glob(str(Path("./") / '**/rb.txt*'), recursive=True)

    documents = SimpleDirectoryReader(input_files=transcript_files).load_data()
    nodes = text_splitter.get_nodes_from_documents(documents)
    # for node in nodes:
    #     print('-' * 100)
    #     print(node.get_content())
    # print(documents)
    # We pass in the service context we instantiated earlier (powered by our open-source LLM)
    index = VectorStoreIndex(nodes, service_context=service_context, show_progress=False)
    # index = VectorStoreIndex.from_documents(nodes, service_context=service_context, show_progress=False)
    query_engine = index.as_query_engine(similarity_top_k=20, node_postprocessors=[rerank])
    # print(query_engine)
    result = query_engine.query("""### FRAGE:
Gibt es Hinweise, dass Eltern oder deren Partner Kinder emotional, psychisch misshandeln? 
Wenn ja, liste diese Hinweise aus dem Kontext auf.

### ERKLÄRUNG:
Mit emotionaler, psychischer Misshandlung ist gemeint: 
Hartnäckiges Muster oder wiederholtes Verhalten der Eltern, das Schaden verursacht beim Kind, wie:
verschmähen, verspotten, feindselig zurückweisen, verhöhnen, ausgrenzen, verunglimpfen, 
Schmerzen zufügen, Fesseln, Einsperren, emotional Gefühle verletzen, Sachen zerstören des Kindes, Kind disziplinieren,
zum Sündenbock machen, für die eigenen Bedürfnisse missbrauchen,
beschuldigen, für die eigenen Fehler verantwortlich machen,
ein negatives Selbstbild durch Beschimpfungen erzeugen, 
erniedrigen, um extreme Enttäuschung und Missbilligung zu erzeugen, 
Leistungen abwerten, 
sich weigern, wechselnde soziale Rollen zu akzeptieren, 
demütigen, terrorisieren, einschüchtern, bedrohen (Verlassenwerden), 
in der Öffentlichkeit lächerlich machen, 
Isolieren: Verhindern, Ängste schüren, weil es soziale Interaktionen wünscht,
Ausbeuten: Verderben des Kindes, Verstärkung, Belohnung von Aggression, 
Ermutigung zu Fehlverhalten, Asozialität, Kriminalität, Hypersexualität, Zwang, sich um die Eltern zu kümmern, 
emotionales Desinteresse gegenüber einem Kind zeigen.

[/INST]"""
                                )    
    # print(result)
    content = []
    scores = []
    for node in result.source_nodes:
        # print(node.id_)
        # print(node.node.get_content()[:])
        # print("reranking score: ", node.score)
        # # print("retrieval score: ", node.node.metadata["retrieval_score"])
        # print("**********")
        cont= 'NODE|' + node.node.get_content()[:] + ' Score: ' + str(node.score) + ' END'
        content.append(cont)
        scores.append(node.score)
    
    file_path = iter+'rb.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
    return result, content, scores
import logging
logging.getLogger().setLevel(logging.ERROR)

def extract_context(text):    
    try:
        extracted_texts = [doc.split("|")[1].split("END',")[0] for doc in text.split("NODE")[1:]]
        # Print the extracted texts (you can save them or process further as needed)
        # print(extracted_texts)
        extracted_context = '\n'.join(extracted_texts)
        extracted_context = '\nChunk:\n'.join(line for line in extracted_context.splitlines() if line.strip())
      #   print(extracted_context)
        return extracted_context
    except IndexError:
        # print("IndexError: list index out of range")
        return None
    

import logging
logging.getLogger().setLevel(logging.ERROR)


answer = []
context = []
scores = []
for index, row in sample_df.iterrows():
    text = row['text']
    antwort, kontext, skores = get_answer(text)
    answer.append(antwort)
    context.append(kontext)
    scores.append(skores)
    

sample_df['antwort'] = answer
sample_df['kontext'] = context
sample_df['scores'] = scores

sample_df.to_excel(iter+'rbs_psych_eltern_sampleTemp2.xlsx')
sample_df0= pd.read_excel(iter+'rbs_psych_eltern_sampleTemp2.xlsx', sheet_name='Sheet1')
sample_df0['extracted_context'] = sample_df0['kontext'].apply(lambda x: extract_context(x))
# sample_df0.to_excel(iter+'rbs_psych_eltern_sampleTemp2.xlsx')
sample_df=sample_df0.copy()

# sample_df.to_excel('rbs_ecan_klient_sample_antwort_llamaIndexRerank1-1000.xlsx')

rbs_alk_eltern_sample=sample_df.copy()

file_path = iter+'rbs_psych_eltern_sampleTemp2.xlsx'
if os.path.exists(file_path):
  os.remove(file_path)



# rbs_alk_eltern_sample_codiert.to_excel('rbs_ecan_klient_sample_antwort_recursFaissRerankBGE500-1000_antwort2.xlsx', index=False)

import sys
# sys.exit()

# dataset = Dataset.from_pandas(rbs_alk_eltern_sample10)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from transformers import LlamaTokenizerFast, AutoTokenizer
import os
model_name ="deepset/gelectra-large"
# Replace this with your own checkpoint
tokenizer = AutoTokenizer.from_pretrained("ecan_klient_class_gelectra", do_lower_case = True,
                                         use_fast=True, max_length=512, truncation=True, padding=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("ecan_klient_class_gelectra").to("cuda")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(device)


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
