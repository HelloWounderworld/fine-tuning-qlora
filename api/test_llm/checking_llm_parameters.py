from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "meta-llama/Llama-2-7b-chat-hf"

#load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#4bitで読み込みたいときは､quantization_configを指定する｡

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            quantization_config=bnb_config,  
                                            device_map="auto",
                                            )

model.print_trainable_parameters()

for name, param in model.named_parameters():
    print(name)

#peftモデルの定義
from peft import LoraConfig, get_peft_model

#adapter層を付けられるレイヤー名
target_modules=[
"embed_tokens",
"lm_head",
"q_proj",
"k_proj",
"v_proj",
"o_proj",
"gate_proj",
"up_proj",
"down_proj",
]

#rはハイパーパラメータ
peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=16, lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
model = get_peft_model(model, peft_config)
