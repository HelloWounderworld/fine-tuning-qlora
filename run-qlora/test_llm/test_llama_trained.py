import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Carregue o tokenizer
tokenizer = LlamaTokenizer.from_pretrained("./output/test_peft")

# Carregue seu modelo fine-tuned
model = LlamaForCausalLM.from_pretrained("./output/test_peft")

# Mova o modelo para a GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Função para gerar texto
def generate_text(prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Teste o modelo
prompt = "Era uma vez"
generated_text = generate_text(prompt)
print(generated_text)