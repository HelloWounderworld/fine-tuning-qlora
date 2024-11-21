import os

model_path = "/caminho/modelo"
required_files = [
    "config.json", 
    "pytorch_model.bin",  # ou model.safetensors
    "tokenizer.json"
]

# Verificar arquivos essenciais
missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

if missing_files:
    print("Arquivos faltando:", missing_files)
else:
    print("Estrutura do modelo parece estar completa!")