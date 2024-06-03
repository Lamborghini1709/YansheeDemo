# -*- coding: utf-8 -*-
import re
import requests
import json
import time
import threading

def make_history(lines_pass):
    ch1 = '甲:'
    ch2 = "乙:"
    history = ''
    for idx, line in enumerate(lines_pass):
        if idx % 2 == 0:
            newline = ch1 + line + '\n'
        else:
            newline = ch2 + line + '\n'
        history += newline
    history += '\n'
    if len(lines_pass) > 0:
        history += '根据甲乙的历史对话,'
    return history

def post_chatglm(prompt):
    global lines_pass
    history = make_history(lines_pass)
    # data = {
    #     "text": history +  "以'甲: " + prompt + "'为开头，生成甲乙间五轮对话",
    # }
    data = {
    "model": "qwen:7b",
    "STREAM":False,
    "messages": [
        {
            "role": "user",
            "content": history +  "以'甲: " + prompt + "'为开头，生成甲乙间五轮对话"
        },
    ]
    }
    print(data)
    r = requests.post(url='http://10.20.94.91:11434/api/chat', json=data, timeout=600)
    # r = requests.post(url='http://120.232.132.10:24003/prompt', data=data)
    # text = r.text
    text = r.json()['message']['content']
    # print(text)

    text = text.replace("：", ":").replace("轮\\w+:", "")
    text = text.replace("（", "(").replace("）", ")")
    list_text_audio = re.split(r'(?:甲:|乙:)', text)
    list_text_audio = [item.strip() for item in list_text_audio if item.strip()]

    print(list_text_audio)
    return list_text_audio

def post_voice_male(url, content):
    data = {
        "interrupt": True,
        "timestamp": 1551838515,
        "tts": content
    }
    headers = {'Content-Type': 'application/json'}
    r = requests.put(url='http://' + url + ':9090/v1/voice/tts', headers=headers, data=json.dumps(data))

def get_tts_state(url):
    data = {
        "timestamp": None
    }
    headers = {'Content-Type': 'application/json'}
    r = requests.get(url='http://' + url + ':9090/v1/voice/tts', headers=headers, data=json.dumps(data))
    res = json.loads(r.text)
    return res['status']

import random
def post_animation(url, op):
    anis = ['raise', 'crouch', 'stretch', 'come on', 'wave', 'bow']
    dirs = ['left', 'right']
    speed = 'slow'
    random.shuffle(anis)
    random.shuffle(dirs)
    data = {
        "motion": {
            "direction": dirs[0],
            "name": anis[0],
            "repeat": 1,
            "speed": speed
        },
        "operation": op,
        "timestamp": 1551838515
    }

    headers = {'Content-Type': 'application/json'}
    r = requests.put(url='http://' + url + ':9090/v1/motions', headers=headers, data=json.dumps(data))

def play_chat(url1, url2):
    global signal, lines_pass, prompts, user_prompt

    for prompt in prompts:
        if signal:
            time.sleep(10)
            continue

        print('正在生成对话，请等待...')
        if user_prompt is not None:
            lines = post_chatglm(user_prompt)
            user_prompt = None
        else:
            lines = post_chatglm(prompt)

        for idx, line in enumerate(lines):
            if signal:
                break
            print(line)
            if idx % 2 == 0:
                post_voice_male(url=url1, content=line)

                while get_tts_state(url=url1) != 'idle':
                    if signal:
                        break
                    # print('wait... ', url1)
                    post_animation(url=url1, op='start')
                    time.sleep(2)
                    post_animation(url=url1, op='stop')

            else:

                post_voice_male(url=url2, content=line)

                while get_tts_state(url=url2) != 'idle':
                    if signal:
                        break
                    # print('wait... ', url2)
                    post_animation(url=url2, op='start')
                    time.sleep(2)
                    post_animation(url=url2, op='stop')

            lines_pass.append(line)
            if len(lines_pass) > 5:
                del lines_pass[0]

def put_asr(url):
    import time
    import re
    headers = {'Content-Type': 'application/json'}
    data = {
        "continues": True,
        "timestamp": 1551838515
        }
    r = requests.put(url='http://' + url + ':9090/v1/voice/asr', headers=headers, data=json.dumps(data))
    # print(r.text)

    question_text = ''
    while True:    
        r = requests.get(url='http://' + url + ':9090/v1/voice/asr', headers=headers)
        data_dict = json.loads(r.text)["data"]

        status = json.loads(r.text)["status"]
        # print(status)
        time.sleep(0.1)

        if len(data_dict) > 1:
            match = re.search(r'{"question":"(.*?)"', data_dict)
            if match:
                prompt = match.group(1)
                # print(prompt)

                if len(prompt) >= 2:
                    if len(prompt) > len(question_text):
                        question_text = prompt
                        continue
                    else:
                        break

        if status == 'idle':
            break

    print(question_text)
    r = requests.delete(url='http://' + url + ':9090/v1/voice/asr', headers=headers)
    print(r.text)
    return question_text

def stopAllOp(url1, url2):
    data = {
        "timestamp": None
    }
    headers = {'Content-Type': 'application/json'}
    r = requests.delete(url='http://' + url1 + ':9090/v1/voice/tts', headers=headers, data=json.dumps(data))
    r = requests.delete(url='http://' + url2 + ':9090/v1/voice/tts', headers=headers, data=json.dumps(data))
    r = requests.delete(url='http://' + url1 + ':9090/v1/motions', headers=headers, data=json.dumps(data))
    r = requests.delete(url='http://' + url2 + ':9090/v1/motions', headers=headers, data=json.dumps(data))
    pass

def getPrompts():
    # with open('prompts.txt', 'r', encoding='utf8') as f:
    with open('input.txt', 'r', encoding='utf8') as f:
        prompts = f.readlines()
        prompts = [item.strip() for item in prompts if item.strip()]
    random.shuffle(prompts)
    return prompts

import os
import random

def main():
    global signal, user_prompt
    url1 = '192.168.2.64' # '192.168.2.138' #'192.168.130.61'
    url2 = '192.168.2.109' # '192.168.2.234'#'192.168.130.154'

    chat_thread = threading.Thread(target=play_chat, args=(url1, url2))
    chat_thread.start()

    while True:
        if user_prompt is not None:
            time.sleep(1)
        user_input = input("请输入命令（a, 语音输入；q，结束）: ")
        if user_input == 'a':
            signal = True
            stopAllOp(url1, url2)
            
            while True:
                prompt = put_asr(url=url1)
                print(prompt)
                if len(prompt) < 2:
                    continue
                break
            
            print(user_prompt)

            user_prompt = prompt
            signal = False

        if user_input == 'q':
            signal = True
            os._exit(0)


if __name__ == '__main__':
    signal = False
    prompts = getPrompts()
    user_prompt = prompts[0]
    lines_pass = []
    main()

    
