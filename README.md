# Understand Fugaku LLM

    https://huggingface.co/Fugaku-LLM/Fugaku-LLM-13B-instruct-gguf

    https://zenn.dev/hellorusk/articles/94bf32ea09ba26

## Creating a virtual environment in Python:

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
17. [LLM JP DPO][17]

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
