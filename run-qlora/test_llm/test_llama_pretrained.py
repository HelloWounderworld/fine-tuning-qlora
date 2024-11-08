import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoConfig


model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
config = AutoConfig.from_pretrained(model_id)
config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

device = "cuda:0"
def ask(text):
  inputs = tokenizer(text, return_tensors="pt").to(device)

  with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))

text = "### Human: 富士山といえば ### Assistant: "
ask(text)
text = "### Human: 明日の天気は ### Assistant: "
ask(text)
text = "### Human: AIといえば ### Assistant: "
ask(text)
