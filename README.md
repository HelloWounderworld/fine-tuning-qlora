# Fine Tuning using qLora at the Fugaku-LLM

    https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct-gguf

    https://zenn.dev/hellorusk/articles/94bf32ea09ba26

## What is Fine tuning?
ファインチューニングはいわゆる人間で例えますと人間の脳を洗脳するようなものです。なのでまずは何をするのかといいますと人間の脳の洗脳を施す前にまずは学習させましてその人の自頭を良くしてなおかつ都合のいいように考え方を偏らせる方法です。

今回はFugaku-LLMと言うもうすであらかじめジェネラルに学習された脳をもっと学習させてそして学習させている最中に自分好みの考え方になるためにその方向性に偏りのある情報で洗脳を施します。

## Steps that I would like to teach to mkae Fine Tuning:

Step 1. Fine Tuning create a ".gguf" file. このファイル制作はいわゆる人間の自頭の作成です。

Step 2. To make Fine Tuning localy without internet.

## Creating a virtual environment in Python and we will install Ollama to run Fugaku LLM:

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

        pip install torch transformers accelerate

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

10. Verify ollama's log


        journalctl -u ollama --no-pager

## Articles that I based to make Fine Tuning:

    https://medium.com/@harsh.vardhan7695/fine-tuning-llama-2-using-lora-and-qlora-a-comprehensive-guide-fd2260f0aa5f

    https://medium.com/@givkashi/fine-tuning-llama-2-model-with-qlora-on-a-custom-dataset-33126b94dee5

    https://note.com/shi3zblog/n/n1a4854ba8949

    https://medium.com/@sdfgh98/gemma2-fine-tuning-from-sft-and-qlora-to-gguf-deployment-with-ollama-3312d1c07ef7

    https://pytorch.org/torchtune/stable/tutorials/qlora_finetune.html

    https://www.kaggle.com/code/philculliton/fine-tuning-with-llama-2-qlora

    https://note.com/kan_hatakeyama/n/n5941dd9d3af4

    https://note.com/npaka/n/nfa56b035c178

    https://note.com/kan_hatakeyama/n/ncd09c52d26c7

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
