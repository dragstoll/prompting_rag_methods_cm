from math import dist
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pandas as pd
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# Load the validated report sample so every subsequent step works on the same slice of data.
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
# sys.exit()

rbs_sample = pd.read_pickle(source)
# sample 40 rows for testing
# rbs_sample = rbs_sample.sample(n=40, random_state=42).reset_index(drop=True)
# Select the checkpoint to run generations with (alternative kept commented for quick switching).
model_name = "QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ"


tokenizer = AutoTokenizer.from_pretrained(model_name,)


# Define stop token ids
stop_token_ids = [0]

PARENT_CONFIGS = {
    "father": {
        "column_prefix": "father",
        "enable_thinking": True,
        "prompt": """
Du sollst die Kooperationsbereitschaft des Vaters bei der Zusammenarbeit mit dem Beistand oder anderen Fachpersonen beurteilen, anhand der unten beschriebenen Definitionen, Vorgaben 
         und basierend auf dem Rechenschaftsbericht und diese anhand folgender Antwortmöglichkeiten abschliessend beurteilen:
         'a: mangelnde Kooperationsbereitschaft'
         'b: Kooperationsbereitschaft vorhanden, hat sich eingestellt'
         'c: keine Hinweise'

         DEFINITIONEN:
         Mangelnde Kooperationsbereitschaft:  Vater befolgt die Anweisung nicht. Vater zeigt sich nicht kooperativ, ist nicht bereit,
         mit dem Beistand oder anderen Fachpersonen zusammenzuarbeiten, reagiert nicht auf Anweisungen des Beistands und anderer involvierter Fachpersonen, 
         befolgt diese nicht, erscheint nicht an vereinbarten Terminen. Es gelingt nicht den Vater zur Perspektivübernahme zu motivieren, fehlende Einsicht des Vaters wird beschrieben. Vater ist nicht bereit mitzuarbeiten.
         Der Vater wird angewiesen / angehalten, wird verpflichtet von der Behörde / Fachpersonen, Weisung Artikel 307 ZGB .

         Kooperationsbereitschaft vorhanden, hat sich eingestellt: Der Vater zeigt sich kooperativ, ist bereit, 
         mit dem Beistand oder anderen Fachpersonen zusammenzuarbeiten oder die Bereitschaft hat sich im Laufe der Zeit eingestellt und wird schlussendlich vorhanden angesehen. 
         Er reagiert auf Anweisungen des Beistands und anderer involvierter Fachpersonen, befolgt diese, erscheint an vereinbarten Terminen.

         

         TIPPS ZUR BEURTEILUNG:
         1. Wenn es Hinweise für sowohl mangelnde Kooperationsbereitschaft beim Vater als auch für sich einstellende Kooperationsbereitschaft gibt,
         dann beurteile die Schilderungen aus dem Rechenschaftsbericht gesamthaft als Verlauf der Berichterstattung, ob es sich die Kooperationsbereitschaft des Vaters im Laufe der Zeit eingestellt hat oder eben nicht.
        2. Wenn es gar keine Hinweise gibt, weder für mangelnde noch für vorhandene, sich einstellende Kooperationsbereitschaft, dann beurteile die Kooperationsbereitschaft als c: keine Hinweise.
         3. Wenn es keine Hinweise auf Kooperationsbereitschaft gibt, dann bedeutet das nicht, dass der Vater nicht kooperativ ist.
         4. Wenn es keine Hinweise auf mangelnde Kooperationsbereitschaft gibt, dann bedeutet das nicht, dass der Vater kooperativ ist.
         5. Wenn es Hinweise über die "Eltern" gibt, dann ist der Vater mitgemeint.


Rechenschaftsbericht:\n\n{report}
""",
    },
    "mother": {
        "column_prefix": "mother",
        "enable_thinking": True,
        "prompt": """
Du sollst die Kooperationsbereitschaft der Mutter bei der Zusammenarbeit mit dem Beistand oder anderen Fachpersonen mit 4000 Tokens beurteilen, anhand der unten beschriebenen Definitionen, Vorgaben 
         und basierend auf dem Rechenschaftsbericht und diese anhand folgender 
         Antwortmöglichkeiten abschliessend beurteilen:
         'a: mangelnde Kooperationsbereitschaft'
         'b: Kooperationsbereitschaft vorhanden, hat sich eingestellt'
         'c: keine Hinweise'

         DEFINITIONEN:
         Mangelnde Kooperationsbereitschaft: Mutter befolgt die Anweisung nicht. Mutter zeigt sich nicht kooperativ, ist nicht bereit,
         mit dem Beistand oder anderen Fachpersonen zusammenzuarbeiten, reagiert nicht auf Anweisungen des Beistands und anderer involvierter Fachpersonen, 
         befolgt diese nicht, erscheint nicht an vereinbarten Terminen. Es gelingt nicht die Mutter zur Perspektivübernahme zu motivieren, fehlende Einsicht der Mutter wird beschrieben. Mutter ist nicht bereit mitzuarbeiten.
         Die Mutter wird angewiesen / angehalten, wird verpflichtet von der Behörde / Fachpersonen, Weisung Artikel 307 ZGB .

         Kooperationsbereitschaft vorhanden, hat sich eingestellt: Die Mutter zeigt sich kooperativ, ist bereit, 
         mit dem Beistand oder anderen Fachpersonen zusammenzuarbeiten oder die Bereitschaft hat sich im Laufe der Zeit eingestellt und wird schlussendlich vorhanden angesehen. 
         Er reagiert auf Anweisungen des Beistands und anderer involvierter Fachpersonen, befolgt diese, erscheint an vereinbarten Terminen.

         

         TIPPS ZUR BEURTEILUNG:
         1. Wenn es Hinweise für sowohl mangelnde Kooperationsbereitschaft bei der Mutter als auch für sich einstellende Kooperationsbereitschaft gibt,
         dann beurteile die Schilderungen aus dem Rechenschaftsbericht gesamthaft als Verlauf der Berichterstattung, ob es sich die Kooperationsbereitschaft der Mutter im Laufe der Zeit eingestellt hat oder eben nicht.
        2. Wenn es gar keine Hinweise gibt, weder für mangelnde noch für vorhandene, sich einstellende Kooperationsbereitschaft, dann beurteile die Kooperationsbereitschaft als c: keine Hinweise.
         3. Wenn es keine Hinweise auf Kooperationsbereitschaft gibt, dann bedeutet das nicht, dass die Mutter nicht kooperativ ist.
         4. Wenn es keine Hinweise auf mangelnde Kooperationsbereitschaft gibt, dann bedeutet das nicht, dass die Mutter kooperativ ist.
         5. Wenn es Hinweise über die "Eltern" gibt, dann ist die Mutter mitgemeint.

Rechenschaftsbericht:\n\n{report}
""",
    },
}
SYSTEM_MESSAGE = "Anweisung: Interpretiere nicht, spekuliere nicht, was sein könnte, sondern beantworte die Fragen anhand der explizit erwähnten Hinweise."

def format_prompt(text, parent):
    """Build a chat prompt from the shared system message plus the parent-specific instructions."""
    cfg = PARENT_CONFIGS[parent]
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": cfg["prompt"].format(report=text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg["enable_thinking"],
    )

# Configure generation and batching settings.
batch_size = 40  # Adjust based on your GPU memory and throughput
parent_order = ("father", "mother")
texts = rbs_sample["text"].tolist()

batched_prompts = []
metadata = []


max_prompt_tokens = 0  # Track the longest prompt to size the model context window safely.
for idx, text in enumerate(texts):
    # Build prompts for each parent so we can run both classifications in one batch.
    for parent in parent_order:
        prompt = format_prompt(text, parent)
        batched_prompts.append(prompt)
        metadata.append((idx, parent))
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)

max_tokens = 8000
model_buffer = 200
computed_max_model_len = max_prompt_tokens + max_tokens + model_buffer
print(f"Max prompt tokens: {max_prompt_tokens}, setting model context length to {computed_max_model_len}")

parent_sampling_params = {
    "father": SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=max_tokens,
    ),
    "mother": SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_tokens=max_tokens,
    ),
}

# Initialize vLLM with conservative multi-GPU settings to avoid memory spikes.
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,   
    #  enforce_eager=True,  # Disable graph capture
    max_num_seqs=20,          # Reduce batch size 
    max_model_len=computed_max_model_len,
    disable_custom_all_reduce=True,  # Reduce multi-GPU overhead
    enable_prefix_caching=True,  # Reduces redundant computation
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
)



# Pre-size storage so we can fill answers row-by-row without reallocations.
results = {
    parent: {
        "antwort": [""] * len(rbs_sample),
        "antwort_thinking": [""] * len(rbs_sample),
        "antwort_content": [""] * len(rbs_sample),
    }
    for parent in parent_order
}

# Execute the generation in batches to respect the max sequence length and GPU memory limits.
for start in range(0, len(batched_prompts), batch_size):
    batch_prompts = batched_prompts[start:start + batch_size]
    batch_params = [
        parent_sampling_params[metadata[start + offset][1]]
        for offset in range(len(batch_prompts))
    ]
    outputs = llm.generate(batch_prompts, batch_params)
    for offset, output in enumerate(outputs):
        row_idx, parent = metadata[start + offset]
        generated_text = output.outputs[0].text
        thinking_content = ""
        content = generated_text
        if "</think>" in generated_text:
            # Split thinking traces from the final answer so both can be stored separately.
            try:
                parts = generated_text.split("</think>", 1)
                thinking_content = parts[0].strip()
                content = parts[1].strip()
            except IndexError:
                content = generated_text
        results[parent]["antwort"][row_idx] = generated_text
        results[parent]["antwort_thinking"][row_idx] = thinking_content
        results[parent]["antwort_content"][row_idx] = content

# Persist the enriched DataFrame (raw answer, thinking, and cleaned content) for downstream analysis.
for parent in parent_order:
    prefix = PARENT_CONFIGS[parent]["column_prefix"]
    rbs_sample[f"{prefix}_antwort"] = results[parent]["antwort"]
    rbs_sample[f"{prefix}_antwort_thinking"] = results[parent]["antwort_thinking"]
    rbs_sample[f"{prefix}_antwort_content"] = results[parent]["antwort_content"]
# Save rbs_sample as an Pikle file
rbs_sample.to_pickle(iter+'rbs_rout1_temp1.pkl')
# Save rbs_sample as an Excel file  
rbs_sample.to_excel(iter+'rbs_rout1_temp1.xlsx', index=False)