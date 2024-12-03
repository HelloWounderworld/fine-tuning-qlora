import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

import locale
def getpreferredencoding(do_setlocale = True):
  return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

from huggingface_hub import snapshot_download
base_model = "./original_model/"
quantized_path = "./quantized_model/"
#
snapshot_download(repo_id=model_id, local_dir=base_model , local_dir_use_symlinks=False)
original_model = quantized_path+'/Llama-2-7b-chat-hf.gguf'