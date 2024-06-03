import pyaudio
import wave
from io import BytesIO
import numpy as np

import webrtcvad
import requests
import time
vad = webrtcvad.Vad()
vad.set_mode(3)


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


def post_wav(audio_data):
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
    print(response.status_code)
    data = response.json()
    text = data['message']
    t2 = time.time()
    print(text, t2-t1)
    
    if len(text)>0:
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
            t3 = time.time()
            print('received_audio', t3-t2)
            # 将响应内容写入文件
            with open('received_audio.wav', 'wb') as f:
                f.write(response.content)
            play_wav('received_audio.wav')
        else:
            print('Failed to retrieve the file')

    

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

        print(absolute_average)
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
                print(len(audio_data))
                post_wav(audio_data)
            lost = 0
            audio_data = b''

except KeyboardInterrupt:
    # 按Ctrl+C退出程序
    pass

finally:
    # 停止并关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("录音结束")
