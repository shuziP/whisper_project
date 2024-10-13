import os
import gradio as gr
import pyaudio
import wave
import whisper
import threading
import json
from zhconv import convert  # 导入 zhconv 库
from llm_call import call_glm 

# 音频录制参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"
JSON_OUTPUT_FILENAME = "recognized_texts.json"
LLM_JSON_OUTPUT_FILENAME = "llm_outputs.json"  # 用于存储大模型输出的JSON文件

audio = pyaudio.PyAudio()

# 录音线程
class Recorder(threading.Thread):
    def __init__(self):
        super().__init__()
        self.is_recording = False

    def run(self):
        self.is_recording = True
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        frames = []

        while self.is_recording:
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

    def stop(self):
        self.is_recording = False

# 使用 Whisper 进行语音识别
def recognize_audio():
    model = whisper.load_model("base")
    result = model.transcribe(WAVE_OUTPUT_FILENAME, language='zh')
    text = result['text']
    
    # 将繁体中文转换为简体中文
    simplified_text = convert(text, 'zh-hans')
    
    return simplified_text

# 将文本追加到 JSON 文件中
def append_to_json(text,FILENAME):
    if not os.path.exists(FILENAME):
        data = []
    else:
        with open(FILENAME, 'r', encoding='utf-8') as f:
            data = json.load(f)

    data.append({"text": text})

    with open(FILENAME, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 从 JSON 文件中获取最近5条记录
def get_recent_history():
    if not os.path.exists(JSON_OUTPUT_FILENAME):
        return "暂无历史数据"

    with open(JSON_OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取最近5条数据并按降序排列
    recent_data = data[-5:][::-1]
    history_text = "\n".join([item['text'] for item in recent_data])
    
    return history_text if history_text else "暂无历史数据"

# 提交用户编辑后的文本
def submit_text(edited_text):
    append_to_json(edited_text,JSON_OUTPUT_FILENAME)  # 将编辑后的文本追加到JSON中
    return get_recent_history()  # 提交后更新历史信息框

def start_recording():
    global recorder
    recorder = Recorder()
    recorder.start()
    return "开始录音..."

def stop_recording():
    recorder.stop()
    recorder.join()
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        return recognize_audio()
    else:
        return "录音文件未找到。"

# 调用大模型并保存输出
def invoke_llm(user_input):
    # 调用 llm_call.py 中的 call_glm 函数
    llm_output = call_glm(user_input=user_input)
    
    # 将大模型的输出保存到 LLM_JSON_OUTPUT_FILENAME 文件中
    append_to_json(llm_output, LLM_JSON_OUTPUT_FILENAME)
    
    return llm_output

# Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        start_btn = gr.Button("对话")
        stop_btn = gr.Button("停止")
        output = gr.Textbox(label="识别结果", interactive=True)  # 设置为可编辑

    submit_btn = gr.Button("提交文本")  # 提交文本按钮
    history_output = gr.Textbox(label="最近五条历史记录", interactive=False)  # 历史记录框，不可编辑

    # 初始化历史信息框
    history_output.update(value=get_recent_history())

    start_btn.click(start_recording, outputs=output)
    stop_btn.click(stop_recording, outputs=output)

    # 点击提交按钮时，更新历史信息框
    submit_btn.click(submit_text, inputs=output, outputs=history_output)

        # 大模型相关
    with gr.Row():
        llm_output = gr.Textbox(label="大模型对话输出", interactive=False)  # 输出框
        llm_btn = gr.Button("大模型对话")  # 大模型对话按钮
    
    # 大模型对话按钮点击逻辑
    llm_btn.click(invoke_llm, inputs=output, outputs=llm_output)

demo.launch()