
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

# rbs_alk_eltern_sample=pd.read_pickle(iter+'sample_dfTemp.pkl')

# file_path = iter + "sample_dfTemp.pkl"
# if os.path.exists(file_path):
#   os.remove(file_path)

# rbs_alk_eltern_sample= pd.read_excel('rbs_ecan_klient_sample_antwort_recursFaissRerankBGE500-1000.xlsx', sheet_name='Sheet1')

import pandas as pd
from transformers import AutoTokenizer
# model_name = "CohereForAI/c4ai-command-r-v01"
model_name = 'CohereForAI/c4ai-command-r-plus-4bit'



# model_config = transformers.AutoConfig.from_pretrained(
#    model_name,
# )

model_id = 'CohereForAI/c4ai-command-r-plus-4bit'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, 
                                        #   add_prefix_space=None
                                          )
# print(tokenizer.default_chat_template)
# template = tokenizer.chat_template
# template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input based on retrieved documents. Do not interpret or speculate.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}



# {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections.' }}{{ '\n\n# System Preamble' }}{{ '\n##ANWEISUNG:' }}{{ '\nBeantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}


# 2: {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\n ## ANWEISUNG: Beantworte die Frage anhand der Erklärung und des vorgegebenen Kontexts, interpretiere oder spekuliere nicht.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{% if citation_mode=='accurate' %}{{ 'Write a response to the user\\'s last input. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}


# 0: {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}{% if citation_mode=='accurate' %}{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}
# tokenizer.push_to_hub(model_name)  # Upload your new template to the Hub!


# tokenizer.default_chat_template = {'default': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}", 'tool_use': '{{ bos_token }}{% if messages[0][\'role\'] == \'system\' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0][\'content\'] %}{% else %}{% set loop_messages = messages %}{% set system_message = \'## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.\' %}{% endif %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' }}{{ \'# Safety Preamble\' }}{{ \'\nThe instructions in this section override those in the task description and style guide sections. Don\\\'t answer questions that are harmful or immoral.\' }}{{ \'\n\n# System Preamble\' }}{{ \'\n## Basic Rules\' }}{{ \'\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\\'s requests, you cite your sources in your answers, according to those instructions.\' }}{{ \'\n\n# User Preamble\' }}{{ \'\n\' + system_message }}{{\'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n\'}}{% for tool in tools %}{% if loop.index0 != 0 %}{{ \'\n\n\'}}{% endif %}{{\'```python\ndef \' + tool.name + \'(\'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \', \'}}{% endif %}{{param_name}}: {% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \'] = None\'}}{% else %}{{ param_fields.type }}{% endif %}{% endfor %}{{ \') -> List[Dict]:\n    """\'}}{{ tool.description }}{% if tool.parameter_definitions|length != 0 %}{{ \'\n\n    Args:\n        \'}}{% for param_name, param_fields in tool.parameter_definitions.items() %}{% if loop.index0 != 0 %}{{ \'\n        \' }}{% endif %}{{ param_name + \' (\'}}{% if not param_fields.required %}{{\'Optional[\' + param_fields.type + \']\'}}{% else %}{{ param_fields.type }}{% endif %}{{ \'): \' + param_fields.description }}{% endfor %}{% endif %}{{ \'\n    """\n    pass\n```\' }}{% endfor %}{{ \'<|END_OF_TURN_TOKEN|>\'}}{% for message in loop_messages %}{% set content = message[\'content\'] %}{% if message[\'role\'] == \'user\' %}{{ \'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'system\' %}{{ \'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>\' + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% elif message[\'role\'] == \'assistant\' %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\'  + content.strip() + \'<|END_OF_TURN_TOKEN|>\' }}{% endif %}{% endfor %}{{\'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\\'Action:\\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        "tool_name": title of the tool in the specification,\n        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>\'}}{% if add_generation_prompt %}{{ \'<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\' }}{% endif %}', 'rag': "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}{% endif %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ '# Safety Preamble' }}{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}{{ '\n\n# System Preamble' }}{{ '\n## Basic Rules' }}{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}{{ '\n\n# User Preamble' }}{{ '\n' + system_message }}{{ '<|END_OF_TURN_TOKEN|>'}}{% for message in loop_messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'system' %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}{{ '<results>' }}{% for document in documents %}{{ '\nDocument: ' }}{{ loop.index0 }}\n{% for key, value in document.items() %}{{ key }}: {{value}}\n{% endfor %}{% endfor %}{{ '</results>'}}{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}{% if citation_mode=='accurate' %}{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}{% endif %}{{ '<|END_OF_TURN_TOKEN|>' }}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}"}
# sys.exit()

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# #################################################################
# # bitsandbytes parameters
# #################################################################

# # Activate 4-bit precision base model loading
# use_4bit = True

# # Compute dtype for 4-bit base models
# bnb_4bit_compute_dtype = "float16"

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"

# # Activate nested quantization for 4-bit base models (double quantization)
# use_nested_quant = False

# #################################################################
# # Set up quantization config
# #################################################################
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=use_4bit,
#    bnb_4bit_quant_type=bnb_4bit_quant_type,
#    bnb_4bit_compute_dtype=compute_dtype,
#    bnb_4bit_use_double_quant=use_nested_quant,
# )

# # Check GPU compatibility with bfloat16
# if compute_dtype == torch.float16 and use_4bit:
#    major, _ = torch.cuda.get_device_capability()
#    if major >= 8:
#        print("=" * 80)
#        print("Your GPU supports bfloat16: accelerate training with bf16=True")
#        print("=" * 80)
       

# #################################################################
# # Load pre-trained config
# #################################################################
# model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    quantization_config=bnb_config,
#    attn_implementation="flash_attention_2",
#    trust_remote_code=True,
# )


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
query = """
## DEFINITION emotionale, psychische Misshandlung der Kinder durch Eltern oder deren Partnern:
Hartnäckiges Muster oder wiederholtes Verhalten der Eltern, das Schaden verursacht beim Kind, wie:
verschmähen, verspotten, feindselig zurückweisen, verhöhnen, ausgrenzen, verunglimpfen, 
Schmerzen zufügen, Fesseln, Einsperren, emotional Gefühle verletzen, Sachen zerstören des Kindes, Kind disziplinieren,
Kind zum Sündenbock machen, Kind für die eigenen Bedürfnisse missbrauchen,
Kind beschuldigen, Kind für die eigenen Fehler verantwortlich machen,
ein negatives Selbstbild durch Beschimpfungen erzeugen, 
erniedrigen, um extreme Enttäuschung und Missbilligung zu erzeugen, 
Leistungen abwerten, 
sich weigern, wechselnde soziale Rollen zu akzeptieren, 
ein Kind demütigen, terrorisieren, einschüchtern, bedrohen (Verlassenwerden), 
in der Öffentlichkeit lächerlich machen. 
Isolieren: Verhindern, Ängste schüren, weil es soziale Interaktionen wünscht. 
Ausbeutung, Verderben des Kindes, Verstärkung, Belohnung von Aggression, 
Ermutigung des Kindes zu Fehlverhalten, Asozialität, Kriminalität, Hypersexualität, Zwang, sich um die Eltern zu kümmern. 
Emotionales Desinteresse gegenüber einem Kind zeigen.
"""
# define conversation input:
conversation = [
    {"role": "user", "content": """
## DEFINITION emotionale, psychische Misshandlung der Kinder durch Eltern oder deren Partnern:
Hartnäckiges Muster oder wiederholtes Verhalten der Eltern, das Schaden verursacht beim Kind, wie:
verschmähen, verspotten, feindselig zurückweisen, verhöhnen, ausgrenzen, verunglimpfen, 
Schmerzen zufügen, Fesseln, Einsperren, emotional Gefühle verletzen, Sachen zerstören des Kindes, Kind disziplinieren,
Kind zum Sündenbock machen, Kind für die eigenen Bedürfnisse missbrauchen,
Kind beschuldigen, Kind für die eigenen Fehler verantwortlich machen,
ein negatives Selbstbild durch Beschimpfungen erzeugen, 
erniedrigen, um extreme Enttäuschung und Missbilligung zu erzeugen, 
Leistungen abwerten, 
sich weigern, wechselnde soziale Rollen zu akzeptieren, 
ein Kind demütigen, terrorisieren, einschüchtern, bedrohen (Verlassenwerden), 
in der Öffentlichkeit lächerlich machen. 
Isolieren: Verhindern, Ängste schüren, weil es soziale Interaktionen wünscht. 
Ausbeutung, Verderben des Kindes, Verstärkung, Belohnung von Aggression, 
Ermutigung des Kindes zu Fehlverhalten, Asozialität, Kriminalität, Hypersexualität, Zwang, sich um die Eltern zu kümmern. 
Emotionales Desinteresse gegenüber einem Kind zeigen.
                                
## NICHT! TEIL DER DEFINITION emotionale, psychische Misshandlung der Kinder durch Eltern oder deren Partnern: 
Auffälliges Verhalten des Kindes, negatives psychisches Verhalten des Kindes, 
verbale oder physische Aggressionen des Kindes gegenüber den Eltern oder anderen Personen,
Suizidgedanken oder Suizidversuche, Ritzen des Kindes usw.                                
                                
## Frage:
Gibt es Hinweise, dass Eltern oder deren Partner die Kinder emotional, psychisch misshandeln, gemäss der DEFINITION oben? 
Wenn ja, liste diese Hinweise auf, anonsten gib an, dass keine Hinweise vorliegen.
"""

# """
     }]

# bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                                            #  quantization_config=bnb_config,
                                              attn_implementation="flash_attention_2",
                                             device_map=("auto")
                                              )


import spacy
from langchain.text_splitter import SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# text_splitter = SpacyTextSplitter(pipeline="de_core_news_lg", chunk_size=1000, chunk_overlap=100)
import re 
import gc

# from FlagEmbedding import FlagReranker
# reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

class Document2:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = ''

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


def get_answer(text):
    with open(iter+'rb.txt', 'w') as file:
        file.write(text)

    loader = TextLoader(iter+"rb.txt")
    # loader = TextLoader(text)
    docs_transformed = loader.load()    
    chunked_documents = text_splitter.split_documents(docs_transformed)
    
    
    db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large'))
    retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 10, },    
    )
    docs = retriever.get_relevant_documents(query)
    # pretty_print_docs(docs)
    list_chunks = [doc.page_content for doc in docs]
    # scores = reranker.compute_score([[query, chunk] for chunk in list_chunks])
    # # print(scores)
    # try:
    #     list_chunks2 = [chunk for _, chunk in sorted(zip(scores, list_chunks), key=lambda x: x[0], reverse=True)]
    # except TypeError:
    #     list_chunks2 = []   
    # # print(list_chunks2)
    # dict_chunks2 = [Document2(text) for text in list_chunks2]
    documents= [doc.page_content for doc in docs]
    # pretty_print_docs(dict_chunks2)
    documents2 = []
    for i, doc in enumerate(documents, start=1):
        documents2.append({
                            # "title": i, 
                           "Kontext": doc})
    # # Print the list
    # print(documents2)
    # for i, chunk in enumerate(extracted_texts, start=1):

    torch.cuda.empty_cache()
    gc.collect()    
    
    # documents2 = documents2[:2]
    # print(documents2)
    # return documents
    # chunked_documents = []
    # for doc in chunked_documents:
    #     chunked_documents.append({"title": doc["title"], "text": doc["text"]})
    # chunked_documents = [
    # { "title": "Tall penguins", "text": "Emperor penguins are the tallest growing up to 122 cm in height." }, 
    # { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
    # ]
    grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
    conversation,
    documents=documents2,
    citation_mode="accurate", #"accurate", # or 
    tokenize=False,
    add_generation_prompt=True,
    return_tensors="pt",
    )
    # print(grounded_generation_prompt)
    input_ids = tokenizer.encode(grounded_generation_prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    gen_tokens = model.generate(
    input_ids, 
    max_new_tokens=512, 
    do_sample=True, 
    temperature=0.1,
    top_k=50, top_p=0.90, 
    repetition_penalty=1.1,
    )
    # gen_text = tokenizer.decode(gen_tokens[0])
    new_tokens = gen_tokens[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(new_tokens)
    # db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large'))
    # retriever = db.as_retriever(
    # search_type="similarity",
    # search_kwargs={'fetch_k': 20, 'k': 10}
    # )


    # rag_chain = (
    # {"context": retriever, "question": RunnablePassthrough()}
    # | llm_chain
    # )
    # result= rag_chain.invoke(query)
    # print(result['text'])
    print(gen_text)
    return gen_text, list_chunks, docs

def extract_context(text):    
    try:
        extracted_texts = [doc.split("page_content='")[1].split("', ")[0] for doc in text.split("Document(")[1:]]
        # Print the extracted texts (you can save them or process further as needed)
        # print(extracted_texts)
        extracted_context = ""
        for i, chunk in enumerate(extracted_texts, start=1):
            extracted_context += f"TEXTABSCHNITT {i}:\n {chunk}\n"
        # print(extracted_context)
        return extracted_context
    except IndexError:
        # print("IndexError: list index out of range")
        return None
    
# context = {}
answer = []
list_chunks1 = []
list_chunks2 = []
for index, row in sample_df.iterrows():
    text = row['text']
    antwort, list1, list2 = get_answer(text)
    answer.append(antwort)
    # context[index] = kontext
    list_chunks1.append(list1)
    list_chunks2.append(list2)
    

sample_df['antwort'] = answer
# sample_df['context'] = context
sample_df['list_chunks1'] = list_chunks1
sample_df['list_chunks2'] = list_chunks2

# # sample_df.to_pickle('sample_dfTemp2.pkl')
# sample_df.to_excel('sample_dfTemp2.xlsx')
# sample_df0= pd.read_excel('sample_dfTemp2.xlsx', sheet_name='Sheet1')
# sample_df0['extracted_context'] = sample_df0['list_chunks2'].apply(lambda x: extract_context(x))
# sample_df0.to_excel('sample_dfTemp2.xlsx')
# sample_df=sample_df0.copy()

# sample_df.to_pickle('commandR_sample_dfTemp.pkl')
# sample_df.to_excel('./psych_klient_classification/commandR_sample_dfTemp.xlsx')
# sample_df.to_excel('rbs_ecan_klient_sample_antwort_llamaIndexRerank1-1000CommandR.xlsx')
sample_df.to_excel(iter+'sample_dfTemp2.xlsx')
sample_df0= pd.read_excel(iter+'sample_dfTemp2.xlsx', sheet_name='Sheet1')
sample_df0['extracted_context'] = sample_df0['list_chunks2'].apply(lambda x: extract_context(x))
# sample_df0.to_excel(iter+'sample_dfTemp2.xlsx')
sample_df=sample_df0.copy()

# sample_df.to_excel('rbs_ecan_klient_sample_antwort_llamaIndexRerank1-1000.xlsx')

rbs_alk_eltern_sample=sample_df.copy()

file_path = iter+'sample_dfTemp2.xlsx'
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
tokenizer = AutoTokenizer.from_pretrained("ecan_klient_class_gelectraCr", do_lower_case = True,
                                         use_fast=True, max_length=512, truncation=True, padding=True,
                                         eos_token='###', pad_token='[PAD]',)
                                          
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("ecan_klient_class_gelectraCr").to("cuda")


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
