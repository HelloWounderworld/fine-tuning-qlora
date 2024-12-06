# Articles that I based to make Fine Tuning:

    https://note.com/kan_hatakeyama/n/ncd09c52d26c7

    https://note.com/kan_hatakeyama/n/n5941dd9d3af4

    https://medium.com/@dillipprasad60/qlora-explained-a-deep-dive-into-parametric-efficient-fine-tuning-in-large-language-models-llms-c1a4794b1766

    https://medium.com/@harsh.vardhan7695/fine-tuning-llama-2-using-lora-and-qlora-a-comprehensive-guide-fd2260f0aa5f

    https://medium.com/@givkashi/fine-tuning-llama-2-model-with-qlora-on-a-custom-dataset-33126b94dee5

    https://note.com/shi3zblog/n/n1a4854ba8949

    https://medium.com/@sdfgh98/gemma2-fine-tuning-from-sft-and-qlora-to-gguf-deployment-with-ollama-3312d1c07ef7

    https://pytorch.org/torchtune/stable/tutorials/qlora_finetune.html

    https://www.kaggle.com/code/philculliton/fine-tuning-with-llama-2-qlora

    https://note.com/kan_hatakeyama/n/n5941dd9d3af4

    https://note.com/npaka/n/nfa56b035c178

    https://note.com/kan_hatakeyama/n/ncd09c52d26c7

    https://stmind.hatenablog.com/entry/2024/09/26/002409

    https://prtimes.jp/main/html/rd/p/000000052.000047565.html

    https://gihyo.jp/article/2024/10/llama3.1-swallow

    https://qiita.com/ryosuke_ohori/items/12c3c6cb6c607fe15d82

    https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795

    https://huggingface.co/blog/mlabonne/sft-llama3

TIP: Is going to be necessary to see python version... I'm using 3.13.0... to much current...

At the conda virtual environment I'm used python3.10.15

1. Check if Nvidia driver and nvidia-cuda-toolkit is installed in root level:

        nvidia-smi

        nvcc --version

1. Configure Ubuntu environment to make possible container's runs:

        https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

1. Docker Proxy configuration:

        https://docs.docker.com/engine/daemon/proxy/

        https://docs.docker.com/engine/cli/proxy/

    Rootless mode (maybe isn't needed)

        https://docs.docker.com/engine/security/rootless/

    Setting proxy

        ~/.config/systemd/user/docker.service.d/http-proxy.conf
        ~/.docker/config.json   

1. Download qLora library using git:

2. Setting CUDA, nvidia-cuda-toolkit, with the same version of your GPU:

        https://hub.docker.com/r/nvidia/cuda

        https://hub.docker.com/layers/nvidia/cuda/12.2.0-runtime-ubuntu20.04/images/sha256-3faf586290da5a86115cbf907f3a34ba48e97875a8e148fa969ddaa6b1472b93

        https://docs.docker.com/compose/how-tos/gpu-support/

        https://github.com/suvash/nixos-nvidia-cuda-python-docker-compose/blob/main/03-nvidia-docker-compose-setup.org

    Root level

        sudo -i

    Check the version

        nvidia-smi

    Check whether there is any nvidia-cuda-toolkit installed

        nvcc --version

    Install nvidia-cuda-toolkit by following link, "nvidia cuda toolkit download 12.x"

        https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

    After that you can install nvidia cuda toolkit

        sudo apt-get install nvidia-cuda-toolkit

    Or there is, and I prefer, another way to make setting this nvidia-cuda-toolkit, using environments variables

        export PATH=/usr/local/cuda/bin:${PATH:+:${PATH}}

        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

2. Check if cuda is working: (Maybe isn't needed)

    Simple test to check whether CUDA/GPU is working

        testing-cuda-gpu

        https://xcat-docs.readthedocs.io/en/stable/advanced/gpu/nvidia/verify_cuda_install.html

        https://rightcode.co.jp/blogs/12323

2. Configure Nvidia-container-toolkit to the docker:

        https://hub.docker.com/r/nvidia/cuda

        https://hub.docker.com/layers/nvidia/cuda/12.2.0-runtime-ubuntu20.04/images/sha256-3faf586290da5a86115cbf907f3a34ba48e97875a8e148fa969ddaa6b1472b93

        https://docs.docker.com/compose/how-tos/gpu-support/

        https://github.com/suvash/nixos-nvidia-cuda-python-docker-compose/blob/main/03-nvidia-docker-compose-setup.org

2. Installing miniconda (talvez nao precise, pois creio que seja a versao do python que permite ou nao a instalacao)

        https://docs.anaconda.com/miniconda/miniconda-install/

        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.
        
        source ~/miniconda3/bin/activate

        conda init --all

        conda create -n meu_ambiente python=3.10
        conda activate gua
        conda install pip

        cd qlora
        pip install -U -r requirements.txt

    Let check if qLora is working, inside qlora, directory (this need setting at the google colab)

        cd examples
        guanaco_7B_demo_colab.ipynb

        pip install scikit-learn --no-build-isolation
        Requirement already satisfied: scikit-learn in /home/teramatsu/miniconda3/envs/gua/lib/python3.13/site-packages (1.5.2)
        Requirement already satisfied: numpy>=1.19.5 in /home/teramatsu/miniconda3/envs/gua/lib/python3.13/site-packages (from scikit-learn) (2.1.2)
        Requirement already satisfied: scipy>=1.6.0 in /home/teramatsu/miniconda3/envs/gua/lib/python3.13/site-packages (from scikit-learn) (1.14.1)
        Requirement already satisfied: joblib>=1.2.0 in /home/teramatsu/miniconda3/envs/gua/lib/python3.13/site-packages (from scikit-learn) (1.4.2)
        Requirement already satisfied: threadpoolctl>=3.1.0 in /home/teramatsu/miniconda3/envs/gua/lib/python3.13/site-packages (from scikit-learn) (3.5.0)

    If necessary

        conda remove --name meu_ambiente --all

2. Install Pyenv:

    We are going to use this service to make possible the installation of a specific python version inside of a virtual environment

        https://ericsysmin.com/2024/01/11/how-to-install-pyenv-on-ubuntu-22-04/

    Definir a Versão Global ou Local:

        pyenv global 3.x.x  # Para definir globalmente
        pyenv local 3.x.x   # Para definir para um projeto específico

    Criar o Ambiente Virtual

        python3.x -m venv nome_do_ambiente

        python3.10 -m venv meu_ambiente

2. Active virtual environment using python

3. To install libraries at the requirements.txt file:

        https://github.com/scikit-learn/scikit-learn/issues/26858

    TIP: https://scikit-learn.org/stable/install.html

4. Testing Llama, before training, using ollama:

        https://zenn.dev/if001/articles/6c507e15cd958b

        https://note.com/npaka/n/n79eebc29366d

        https://ollama.com/blog/run-llama2-uncensored-locally

        https://www.silasreinagel.com/blog/ai/llama2/llm/2024/03/14/ai-how-to-run-llama-2-locally/

4. Setting parameters to train:

    You have to build an access token

        https://huggingface.co/docs/hub/security-tokens

    If you are using any shared stored space, you can set cache_dir to organize LLM's that you are donwloading

        https://github.com/vllm-project/vllm/issues/764

    Download, in first place, the Fugaku.

    At this link, https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct-gguf, you have to download

        Fugaku-LLM-13B-instruct-0325b-q5_k_m.gguf

    And at this link, https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B/tree/main, you have to download the following (Nao sei se foi util, tentar fazer isso com o comando, sh scripts/finetune_guanaco_65b.sh)

        config.json
        pytorch_model.bin.index.json
        special_tokens_map.json
        tokenizer_config.json
        tokenizer.json
        pytorch_model-00001-of-00006.bin
        pytorch_model-00002-of-00006.bin
        pytorch_model-00003-of-00006.bin
        pytorch_model-00004-of-00006.bin
        pytorch_model-00005-of-00006.bin
        pytorch_model-00006-of-00006.bin

    And at this link, https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct/tree/main, you have to download the following

        config.json
        generation_config.json
        special_tokens_map.json
        tokenizer_config.json
        tokenizer.json
        model.safetensors.index.json
        trainer_state.json
        training_args.bin
        model-00001-of-00006.safetensors
        model-00002-of-00006.safetensors
        model-00003-of-00006.safetensors
        model-00004-of-00006.safetensors
        model-00005-of-00006.safetensors
        model-00006-of-00006.safetensors

    You have to set access token to make download with qlora.py.... where can I set this access token?

    Maybe I have to make install, using pip, hugging-face cli to make login by this token

        https://huggingface.co/docs/huggingface_hub/guides/cli

    And put out all these files inside model, directory

        python qlora.py –learning_rate 0.0001 --model_name_or_path Fugaku-LLM/Fugaku-LLM-13B-instruct-0325b-q5_k_m.gguf \
            --use_auth_token 
            --output_dir ./result/test_peft

    or

        sh scripts/finetune_guanaco_65b.sh 

    For models larger than 13B, we recommend adjusting the learning rate:

        python qlora.py –learning_rate 0.0001 --model_name_or_path <path_or_name>

        python qlora.py \
            --model_name model/Fugaku-LLM-13B-instruct-0325b-q5_k_m.gguf \
            --output_dir ./output/test_peft \
            --dataset_name shi3z/anthropic_hh_rlhf_japanese\
            --max_steps 1000 \
            --use_auth \
            --logging_steps 10 \
            --save_strategy steps \
            --data_seed 42 \
            --save_steps 50 \
            --save_total_limit 40 \
            --max_new_tokens 32 \
            --dataloader_num_workers 1 \
            --group_by_length \
            --logging_strategy steps \
            --remove_unused_columns False \
            --do_train \
            --lora_r 64 \
            --lora_alpha 16 \
            --lora_modules all \
            --double_quant \
            --quant_type nf4 \
            --bf16 \
            --bits 4 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type constant \
            --gradient_checkpointing \
            --dataset hh-rlhf \
            --source_max_len 16 \
            --target_max_len 512 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 16 \
            --eval_steps 187 \
            --learning_rate 0.0002 \
            --adam_beta2 0.999 \
            --max_grad_norm 0.3 \
            --lora_dropout 0.1 \
            --weight_decay 0.0 \
            --seed 0 \
            --load_in_4bit \
            --use_peft \
            --batch_size 4 \
            --gradient_accumulation_steps 2 \
            --output_dir peft_test

    Making fine tuning with Llama2

        python qlora.py \
            --model_name meta-llama/Llama-2-7b-hf \
            --output_dir ./output/test_peft \
            --dataset_name shi3z/anthropic_hh_rlhf_japanese\
            --max_steps 1000 \
            --use_auth \
            --logging_steps 10 \
            --save_strategy steps \
            --data_seed 42 \
            --save_steps 50 \
            --save_total_limit 40 \
            --max_new_tokens 32 \
            --dataloader_num_workers 1 \
            --group_by_length \
            --logging_strategy steps \
            --remove_unused_columns False \
            --do_train \
            --lora_r 64 \
            --lora_alpha 16 \
            --lora_modules all \
            --double_quant \
            --quant_type nf4 \
            --bf16 \
            --bits 4 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type constant \
            --gradient_checkpointing \
            --dataset hh-rlhf \
            --source_max_len 16 \
            --target_max_len 512 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 16 \
            --eval_steps 187 \
            --learning_rate 0.0002 \
            --adam_beta2 0.999 \
            --max_grad_norm 0.3 \
            --lora_dropout 0.1 \
            --weight_decay 0.0 \
            --seed 0 \
            --load_in_4bit \
            --use_peft \
            --batch_size 4 \
            --gradient_accumulation_steps 2

        python qlora.py \
            --model_name meta-llama/Llama-2-7b-chat-hf \
            --output_dir ./output/test\
            --dataset test_llm/dataset/json/test.json \
            --dataset_format input-output\
            --max_steps 1000 \
            --use_auth \
            --logging_steps 10 \
            --save_strategy steps \
            --data_seed 42 \
            --save_steps 50 \
            --save_total_limit 40 \
            --dataloader_num_workers 1 \
            --group_by_length \
            --logging_strategy steps \
            --remove_unused_columns False \
            --do_train \
            --lora_r 64 \
            --lora_alpha 16 \
            --lora_modules all \
            --double_quant \
            --quant_type nf4 \
            --bf16 \
            --bits 4 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type constant \
            --gradient_checkpointing \
            --source_max_len 16 \
            --target_max_len 4096 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 16 \
            --eval_steps 187 \
            --learning_rate 0.0002 \
            --adam_beta2 0.999 \
            --max_grad_norm 0.3 \
            --lora_dropout 0.1 \
            --weight_decay 0.0 \
            --seed 0 \
            --load_in_4bit \
            --use_peft \
            --batch_size 4 \
            --gradient_accumulation_steps 2

    Where the LLM is downloaded, inside docker container, if you build it as root level

        /root/.cache/huggingface/hub

5. Converting Fine-Tuned model to the .gguf file - to make more easy to test with ollama:

        https://huggingface.co/docs/transformers/main/gguf

        https://github.com/ggerganov/llama.cpp.git

        https://qiita.com/hudebakononaka/items/ca295eae60231d7d025f

        https://qiita.com/kyotoman/items/5d460708d798a6f26cf0

        https://www.reddit.com/r/LocalLLaMA/comments/1amjx77/how_to_convert_my_finetuned_model_to_gguf/?rdt=56046

        https://github.com/ggerganov/llama.cpp/discussions/4997

        https://github.com/ggerganov/llama.cpp/discussions/2948

        https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987

    Make git clone of llama.cpp

        GGML_CUDA=1 make

        pip install -r requirements/requirements-convert-hf-to-gguf.txt

    

6. TIP: If your home directory storage is almost filled, you could change the place where you can storage following instructions below

        https://community.sisense.com/t5/knowledge-base/relocating-var-lib-docker-directory/ta-p/18596

        https://forums.docker.com/t/change-the-default-docker-storage-location/140455

        https://evodify.com/change-docker-storage-location/

        https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux

        https://stackoverflow.com/questions/59345566/move-docker-volume-to-different-partition

        https://forums.docker.com/t/how-to-change-var-lib-docker-directory-with-overlay2/43620