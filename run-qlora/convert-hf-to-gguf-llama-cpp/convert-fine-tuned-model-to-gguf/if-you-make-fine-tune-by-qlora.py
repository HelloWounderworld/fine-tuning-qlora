from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Carrega o modelo base
base_model_name = "nome_do_modelo_base"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Carrega os adaptadores
peft_model_path = "/caminho/para/output"
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Merge os pesos
merged_model = model.merge_and_unload()

# Salva o modelo merged
merged_model.save_pretrained("/caminho/para/modelo_merged")
base_tokenizer.save_pretrained("/caminho/para/modelo_merged")
