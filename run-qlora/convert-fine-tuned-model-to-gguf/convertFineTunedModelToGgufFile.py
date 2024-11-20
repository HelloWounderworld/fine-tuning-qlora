from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Salva o modelo e tokenizer
tokenizer.save_pretrained('./llm-pretrained/')
model.save_pretrained('./llm-model/')


from llama_cpp import Llama

filename = "Llama-2-7b-chat-hf-v1.0.Q6_K.gguf"

# Carrega o modelo
llm = Llama(
    model_path=filename,
    n_ctx=2048,  # tamanho do contexto
    n_threads=4  # número de threads
)

# Exemplo de uso
output = llm(
    "What is Python?",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95
)
