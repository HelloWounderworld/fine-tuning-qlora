# Fine Tuning using qLora

    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

    https://huggingface.co/meta-llama/Llama-2-7b-hf

    https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct/tree/main

## What is Fine tuning?
ファインチューニングはいわゆる人間で例えますと人間の脳をよく言えば教育で悪く言えば洗脳するようなものです。なのでまずは何をするのかといいますと人間の脳の洗脳を施す前にまずは学習させましてその人の自頭を良くしてなおかつ都合のいいように考え方を偏らせる方法です。

今回はFugaku-LLMと言うもうすであらかじめジェネラルに学習された脳をもっと学習させてそして学習させている最中に自分好みの考え方になるためにその方向性に偏りのある情報で洗脳を施します。

## Hyperparameters in LLM's and Fine-Tuning tools
An introductory books to understand about this concept are

    Pattern Recognition and Machine Learning - Christopher M. Bishop

    Deep Learning - Christopher M. Bishop

    Deep Learning - Ian GoodFellow, Yoshua Bengio and Aaron Courville

### Hyperparameters in the Machine Learning

    https://medium.com/@ompramod9921/model-parameters-and-hyperparameters-in-machine-learning-502799f982d7 - Read

    https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/ - Read

    https://datascience.stackexchange.com/questions/14187/what-is-the-difference-between-model-hyperparameters-and-model-parameters - Read

    https://datascience.stackexchange.com/questions/31614/which-parameters-are-hyper-parameters-in-a-linear-regression - Read

    https://towardsdatascience.com/parameters-and-hyperparameters-aa609601a9ac

    https://www.analyticsvidhya.com/blog/2022/02/a-comprehensive-guide-on-hyperparameter-tuning-and-its-techniques/

    https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide

    https://www.ibm.com/think/topics/hyperparameter-tuning

    https://www.reddit.com/r/LocalLLaMA/comments/1aq1u3n/qlora_hyperparameters_for_small_finetuning_task/

    https://www.entrypointai.com/blog/lora-fine-tuning/

    https://dl.acm.org/doi/10.1145/3533767.3534400

    https://www.researchgate.net/publication/343390531_On_Hyperparameter_Optimization_of_Machine_Learning_Algorithms_Theory_and_Practice

    https://arxiv.org/abs/2312.04528

    https://www.cambridge.org/core/journals/political-science-research-and-methods/article/role-of-hyperparameters-in-machine-learning-models-and-how-to-tune-them/27296C04CF5935C55327F11BF4017371

### Hyperparameters in the fine-tuning

    https://encord.com/blog/fine-tuning-models-hyperparameter-optimization/s

    https://www.entrypointai.com/blog/fine-tuning-hyperparameters/

## BLEU metric to measure how smart the LLM became

    https://cloud.google.com/translate/docs/advanced/automl-evaluate?hl=pt-br

Throught of a dataset, in this time, we will considere the 10% of this data as a reference to measure how smart the LLM became.

## Steps that I would like to teach to make Fine Tuning:

Step 1. Fine Tuning create a ".gguf" file. このファイル制作はいわゆる人間の自頭の作成です。

Step 2. To make Fine Tuning localy without internet.

## Docker General Settings:

### Docker Proxy configuration:

    https://docs.docker.com/engine/daemon/proxy/

    https://docs.docker.com/engine/cli/proxy/

## Step by Setp to make fine-tuning:

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

We have a lot of steps to finally make possible to begin fine-tuning

### Setting up the environment to make it possible to install NVIDIA and CUDA within the container:

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

2. Configure Nvidia-container-toolkit to the docker:

        https://hub.docker.com/r/nvidia/cuda

        https://hub.docker.com/layers/nvidia/cuda/12.2.0-runtime-ubuntu20.04/images/sha256-3faf586290da5a86115cbf907f3a34ba48e97875a8e148fa969ddaa6b1472b93

        https://docs.docker.com/compose/how-tos/gpu-support/

        https://github.com/suvash/nixos-nvidia-cuda-python-docker-compose/blob/main/03-nvidia-docker-compose-setup.org

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

### Building Docker container and to make fine-tuning inside:

1. Download qLora library using git:

2. Login at the Huggingface-cli:

    huggingface-cli --help

    huggingface-cli login

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

5. To make Fine-tuning using Alpaca format:

        https://zackproser.com/blog/how-to-create-a-custom-alpaca-dataset

        https://note.com/npaka/n/n1a0ab681dc70

        https://www.mlexpert.io/blog/alpaca-fine-tuning

        https://note.com/npaka/n/na3f5abf30629

5. To make merge:

        https://adapterhub.ml/blog/2024/08/adapters-update-reft-qlora-merging-models/

6. TIP: If your home directory storage is almost filled, you could change the place where you can storage following instructions below

        https://community.sisense.com/t5/knowledge-base/relocating-var-lib-docker-directory/ta-p/18596

        https://forums.docker.com/t/change-the-default-docker-storage-location/140455

        https://evodify.com/change-docker-storage-location/

        https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux

        https://stackoverflow.com/questions/59345566/move-docker-volume-to-different-partition

        https://forums.docker.com/t/how-to-change-var-lib-docker-directory-with-overlay2/43620

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
