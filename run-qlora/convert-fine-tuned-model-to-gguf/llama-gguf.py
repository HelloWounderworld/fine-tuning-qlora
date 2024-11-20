from llama_cpp import Llama

# Caminho para seu arquivo GGUF
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

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