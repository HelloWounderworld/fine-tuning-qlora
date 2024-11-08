# Fine Tuning using qLora at the Fugaku-LLM

    https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct-gguf

    https://zenn.dev/hellorusk/articles/94bf32ea09ba26

## What is Fine tuning?
ファインチューニングはいわゆる人間で例えますと人間の脳をよく言えば教育で悪く言えば洗脳するようなものです。なのでまずは何をするのかといいますと人間の脳の洗脳を施す前にまずは学習させましてその人の自頭を良くしてなおかつ都合のいいように考え方を偏らせる方法です。

今回はFugaku-LLMと言うもうすであらかじめジェネラルに学習された脳をもっと学習させてそして学習させている最中に自分好みの考え方になるためにその方向性に偏りのある情報で洗脳を施します。

## Steps that I would like to teach to mkae Fine Tuning:

Step 1. Fine Tuning create a ".gguf" file. このファイル制作はいわゆる人間の自頭の作成です。

Step 2. To make Fine Tuning localy without internet.

## Creating a virtual environment in Python and we will install Ollama to run Fugaku LLM:
Article where I based

    https://tech.takuyakobayashi.jp/2024/05/18/23#google_vignette

1. Checking python version:

        python --version or python3 --version

2. Checking if venv installed:

    Usually, Python 3.3 to above the venv comes together.

         python -m venv --help

    If not just install running the following

        sudo apt-get install python3.x-venv

3. Creating an virtual environment:

    Choose the directory that you want to develop and digit the following command

            python -m venv environment_name

4. Activating the virtual environment:

    Window:

         environment_name\Scripts\activate

    MacOs/Linux:

         source environment_name/bin/activate

5. Now, you can install, using pip, fastapi package:

    Before to install packages

        python -m pip install --upgrade pip

    And now, you can install

        pip install -q accelerate==0.26.1 peft==0.4.0 bitsandbytes==0.42.0 transformers==4.36.2 trl==0.4.7 datasets==2.16.1

    If you have a requirements.txt, just copy it inside of virtual environment directory and run following command

         pip install -r requirements.txt

6. Freeze packages versions on the requirements.txt file:

    After installed packages that you need to your project is a good practice to freeze its in a requirements.txt files.

    To do it, you have, in first, digit

        pip freeze

    After this you have to create a requirements file and to make Ctrl+C and Ctrl+V about this file. Or, more fast

        pip freeze > requirements.txt

7. (Tip) If you want to get out of the virtual environment just type:

        deactivate

8. Installing Ollama:

        sudo apt-get update

        sudo apt-get install curl

        curl -fsSL https://ollama.com/install.sh | sh

9. Run Ollama:

    In first place, you have to in at the model directory

        cd model

    In the following to run the command below to read modelfile.txt

        ollama create fugaku -f modelfile.txt

    After this, you can start the ollama

        ollama run fugaku

    TIP: If you need to finish the chat, you just need to type

        /bye

10. Verify ollama's log


        journalctl -u ollama --no-pager

## Articles that I based to make Fine Tuning:

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

4. Testing Llama, before training:

    https://zenn.dev/if001/articles/6c507e15cd958b

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
            --gradient_accumulation_steps 2 \
            --output_dir peft_test

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

5. Initialize training:

## Creating an Interactive Chat by Ollama:

    https://github.com/ollama-ui/ollama-ui

    https://tech.takuyakobayashi.jp/2024/05/18/23#google_vignette

## References:

1. [Fugaku LLM ChatBot][1]
2. [Fugaku-LLM DeepSpeedFugaku][2]
3. [Pytorch Nvidia][3]
4. [Awesome Japanese LLM][4]
5. [Nvidia Driver Download][5]
6. [Nvidia Container Toolkit][6]
7. [Nvidia Container Toolkit Documentation][7]
8. [Ollama Instructions to Use Fugaku-LLM 13B GGUF][8]
9. [Ollama Instructions to Use Fugaku-LLM 13B GGUF by Ubuntu][9]
10. [Fugaku LLM ChatBot How to Build][10]
11. [Fugaku LLM ChatBot How to Build 2][11]
12. [Fugaku LLM ChatBot Ollama Chat Interface][12]
13. [Fugaku LLM 13B Japanese language model][13]
14. [How to training an LLM using Fine Tune by Transformers Reforiciment Learning STF][14]
15. [LLM JP STF Fine Tuning][15]
16. [LLM JP Tokenizer][16]
17. [LLM JP DPO - This uses many GPU Memory][17]
18. [Lora Fine tuning][18]
19. [qLora Fine tuning][19]
20. [qLora Fine tuning article][20]
21. [Phi-3][21]
22. [Language Models are Few-Shot Learners - Foundation][22]
23. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models - Kind of Prompt Engineering][23]
24. [Phi-3 - FineTuning][24]
25. [Ollama modelfile.txt parameters settings][25]
26. [Curiosity about Content Creation using LLM Open Source Model][26]
27. [Clustering][27]
28. [Docker Hub Nvidia/CUDA][28]
29. [Docker Hub Nvida CUDA 12.2][29]
30. [Setting Nvida Docker Compose][30]
31. [Setting Nvidia Docker compose Github][31]
32. [Nvidia Container Toolkit][32]
33. [Setting proxy at the .bashrc][33]
34. [Setting proxy at the .bashrc][34]
35. [Simple test whether CUDA/GPU is working using Pytorch][35]

[1]: https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct
[2]: https://github.com/Fugaku-LLM/DeepSpeedFugaku
[3]: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
[4]: https://github.com/llm-jp/awesome-japanese-llm
[5]: https://www.nvidia.com/content/DriverDownloads/confirmation.php?url=/Windows/531.61/531.61-desktop-win10-win11-64bit-international-dch-whql.exe&lang=us&type=GeForce
[6]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.4/release-notes.html
[7]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.4/index.html
[8]: https://zenn.dev/hellorusk/articles/94bf32ea09ba26
[9]: https://tech.takuyakobayashi.jp/2024/05/18/23
[10]: https://note.com/owlet_notes/n/nd144bd2d1dc1
[11]: https://note.com/ngc_shj/n/n7a8ce01f13ac
[12]: https://github.com/ollama-ui/ollama-ui
[13]: https://dataloop.ai/library/model/fugaku-llm_fugaku-llm-13b/
[14]: https://huggingface.co/docs/trl/index
[15]: https://github.com/llm-jp/llm-jp-sft
[16]: https://github.com/llm-jp/llm-jp-tokenizer
[17]: https://github.com/llm-jp/llm-jp-dpo
[18]: https://github.com/microsoft/LoRA
[19]: https://github.com/artidoro/qlora
[20]: https://arxiv.org/abs/2305.14314
[21]: https://github.com/microsoft/Phi-3CookBook
[22]: https://arxiv.org/abs/2005.14165
[23]: https://arxiv.org/abs/2201.11903
[24]: https://zenn.dev/headwaters/articles/55f648399c1820
[25]: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter
[26]: https://byrayray.medium.com/llama-3-2-vs-llama-3-1-vs-gemma-2-finding-the-best-open-source-llm-for-content-creation-1f6085c9f87a
[27]: https://www.ibm.com/topics/clustering
[28]: https://hub.docker.com/r/nvidia/cuda
[29]: https://hub.docker.com/layers/nvidia/cuda/12.2.0-runtime-ubuntu20.04/images/sha256-3faf586290da5a86115cbf907f3a34ba48e97875a8e148fa969ddaa6b1472b93
[30]: https://docs.docker.com/compose/how-tos/gpu-support/
[31]: https://github.com/suvash/nixos-nvidia-cuda-python-docker-compose/blob/main/03-nvidia-docker-compose-setup.org
[32]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[33]: https://qiita.com/Fal318/items/338521feb42197a3aee5
[34]: https://qiita.com/nullsnet/items/66590b0ff33e15db7532
[35]: https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
