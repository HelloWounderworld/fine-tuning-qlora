from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Salva o modelo e tokenizer
tokenizer.save_pretrained('./modelbase/')
model.save_pretrained('./modelbase/')
