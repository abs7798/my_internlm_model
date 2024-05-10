import gradio as gr
import os

# download internlm2 to the base_path directory using git tool
base_path = './my_internlm_model'
os.system(f'mkdir my_internlm_model')

print(f'创建文件夹{os.path.isdir(base_path)}')
os.system(f'cd {base_path}')
os.system("git lfs install")
os.system(f'git clone https://code.openxlab.org.cn/abs7798/my_internlm_model.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system("pip install sentencepiece")
os.system("pip install einops")

'''from lmdeploy import pipeline, TurbomindEngineConfig
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2) 

pipe = pipeline(base_path, backend_config=backend_config)

def model(image, text):
    response = pipe((text, image)).text
    return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Textbox(),], outputs=gr.Chatbot())
demo.launch()  '''


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True)

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-7B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
