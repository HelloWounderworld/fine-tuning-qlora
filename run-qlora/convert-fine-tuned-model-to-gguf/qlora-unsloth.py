from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("lora_model")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method = "q4_k_m")