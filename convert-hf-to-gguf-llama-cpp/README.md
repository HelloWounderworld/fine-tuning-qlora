# Step by Step how to make the convertion from hf to the gguf

    https://adapterhub.ml/blog/2024/08/adapters-update-reft-qlora-merging-models/

## Convert modelfile to .gguf

    python llama.cpp/convert_hf_to_gguf.py ./convert-hf-to-gguf/merged_model_qlora_peft --outtype q8_0 --outfile ./gguf-file-complete/Llama-2-7b-chat-hf-fine-tuning-peft-q8_0.gguf
