# Primeiro, instale a biblioteca transformers se ainda não tiver feito isso:
# pip install transformers torch

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Carregue o tokenizer e o modelo pré-treinado
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

# Mova o modelo para a GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Texto de entrada
text = "Era uma vez"

# Tokenize o texto e mova os tensores para o dispositivo
inputs = tokenizer(text, return_tensors="pt").to(device)

# Geração de texto
with torch.no_grad():  # Desativa o cálculo de gradientes
    outputs = model.generate(**inputs, max_new_tokens=50)  # Gera até 50 novos tokens

# Decodifique os IDs de volta para texto
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Mostre o texto gerado
print(generated_text)