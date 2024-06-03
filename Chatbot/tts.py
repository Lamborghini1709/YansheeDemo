import pyaudio
import wave
from io import BytesIO
import numpy as np

# import webrtcvad
import requests
import time
# vad = webrtcvad.Vad()
# vad.set_mode(3)

import string
def is_punctuation(char):
    punctuation = string.punctuation + ",，。!！?？；;：:、'‘’“”\"（）()[]【】<>《》〈〉～·"
    return char in punctuation

# from zhipuai import ZhipuAI
def get_chatglm_api(prompt):
    client = ZhipuAI(api_key="bc9dde0bad95f5d9ce49d091353c9416.iRbtvffz9h7hLcup") # 填写您自己的APIKey
    text = '抱歉，我没有理解您的意思。请换一种方式提问。'
    try:
        response = client.chat.completions.create(
            model="glm-3-turbo",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        # text = str(response.choices[0].message.content)
        return response
    except Exception as e:
        print('发生错误：' + str(e))
    return text

def get_chatglm_local(prompt):
    url = 'http://39a038269a.qicp.vip/chat'
    # url = '10.20.19.51:12003/chat'
    data = { 'query': prompt, 'clear': 'false' }
    text = '抱歉，我没有理解您的意思。请换一种方式提问。'
    try:
        response = requests.post(url, data=data, stream=True)
        response.raise_for_status()
        return response
    except Exception as e:
        print('发生错误：' + str(e))
    return text

# 定义录音参数
CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 数据格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率
RECORD_SECONDS = 2  # 录音持续时间（秒）
chunk_size = 200

def play_wav(wavname):
    chunk = 1024
    wf = wave.open(wavname, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    data = wf.readframes(chunk)
    
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    
    time.sleep(0.5)

    stream.stop_stream()
    stream.close()
    
    p.terminate()

def play_text():
    global playing, speaking, wavlist
    while playing:
        if len(wavlist) == 0:
            speaking = False
            time.sleep(0.1)
        else:
            speaking = True
            wavcontent = wavlist.pop(0)
            with open('received_audio.wav', 'wb') as f:
                f.write(wavcontent)
            play_wav('received_audio.wav')

def post_tts(text):
    # 假设你有一个可以返回音频文件的URL
    url = 'http://120.232.132.10:24001/audio'

    # 要发送的数据
    data = {
        'text': text
    }

    # 发送POST请求
    response = requests.post(url, data=data)

    # 检查响应状态码
    if response.status_code == 200:
        return response.content
    else:
        print('Failed to retrieve the file')
    
    return None

def post_wav(audio_data):
    global wavlist
    t1 = time.time()
    # 创建BytesIO对象并写入WAV数据
    speak_data = BytesIO()
    with wave.open(speak_data, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 16位采样，每个采样占用2字节
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_data)

    # 重置文件指针到开始位置
    speak_data.seek(0)

    # 指定要POST的URL
    url = 'http://120.232.132.10:24001/upload'

    # 准备文件数据
    files = {'file': ('audio.wav', speak_data, 'audio/wav')}

    # 发送POST请求，将speak_data作为文件发送
    response = requests.post(url, files=files)

    # 打印响应内容
    # print(response.status_code)
    data = response.json()
    text = data['message']
    t2 = time.time()
    print(text, t2-t1)
    
    if len(text)>0:
        response = get_chatglm_local(text)
        t3 = time.time()
        print('chatglm_api', t3-t2)
        
        if type(response) == str:
            text = response
        else:
            newtext = ''
            for chunk in response.iter_content(chunk_size=8192):
                # onetoken = chunk.choices[0].delta.content
                onetoken = chunk.decode('utf-8')
                print(onetoken)
                newtext += onetoken
                if len(newtext)>10 and is_punctuation(onetoken):
                    # play_text(newtext)
                    # print(newtext)
                    wavlist.append(post_tts(newtext))
                    newtext = ''
                    time.sleep(0.1)
            if len(newtext)>0:
                wavlist.append(post_tts(newtext))
                newtext = ''
                time.sleep(0.1)

def chat():
    global playing, speaking
    # 初始化PyAudio实例
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("开始录音...")

    import struct

    try:
        audio_data = b''
        lost = 0
        begin = False
        while True:
            # print('speaking', speaking)
            if speaking:
                time.sleep(0.1)
                continue
            try:
                frame = stream.read(CHUNK)
                # 将每个字节转换为整数
            except Exception as e:
                continue

            integers = []
            for i in range(0, len(frame), 2):
                int_value = struct.unpack('<h', frame[i:i+2])[0]
                integers.append(abs(int_value))

            absolute_sum = sum(integers)
            absolute_average = absolute_sum / len(integers)

            #　print(absolute_average)
            if absolute_average > 75:
                begin = True
                lost = 0
            else:
                lost += 1

            if begin:
                audio_data += frame

            if lost > 10:
                begin = False
                if len(audio_data)>4096:
                    # print(len(audio_data))
                    post_wav(audio_data)
                lost = 0
                audio_data = b''

    except KeyboardInterrupt:
        # 按Ctrl+C退出程序
        playing = False
        pass

    finally:
        # 停止并关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("录音结束")

import threading
if __name__ == '__main__':
    playing = True
    speaking = False
    wavlist = []
    my_thread = threading.Thread(target=play_text)
    my_thread.start()
    
    chat()
