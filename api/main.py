#!/bin/python3
import readline #input()を使う時は必須！
import sys
import json
import ollama
from datetime import datetime

modelfile='''
FROM fugaku
PARAMETER temperature 1
PARAMETER num_predict -2
PARAMETER num_ctx 4096
'''
ollama.create(model='aichat', modelfile=modelfile)

def ai_thinking(messages):
    response = ollama.chat(
        model = "aichat",
        messages = messages,
        stream = True, #パラパラッと表示する
    )
    print("\n")
    response_text = ''
    for chunk in response:
        response_text += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    print("\n")
    return response_text

def ai_session(messages):
    while True:
        user_input = input(">>> ")
        if user_input.lower() == 'exit':
            break

        messages.append({"role": "user", "content": user_input})
        
        ai_output = ai_thinking(messages)

        messages.append({"role": "assistant", "content": ai_output})

    with open(logfile, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)

def load_messages_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

if len(sys.argv) > 1:
    logfile = sys.argv[1]
else:
    now = datetime.now()
    logfile = "aichat-"+now.strftime('%Y%m%d%H%M%S')+".log"

messages = load_messages_from_file(logfile)

if not messages:
    messages = [
        {
            "role": "system",
            "content": "ここは自由なチャットルームです。",
        },
    ]

ai_session(messages)
