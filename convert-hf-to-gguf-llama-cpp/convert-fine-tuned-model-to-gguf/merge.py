from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Carrega o modelo base
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Carrega os adaptadores
peft_model_path = "./../output/test_peft/checkpoint-1000/"
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Merge os pesos
merged_model = model.merge_and_unload()

# Salva o modelo merged
merged_model.save_pretrained("./merged_model_qlora_peft")
base_tokenizer.save_pretrained("./merged_model_qlora_peft")
