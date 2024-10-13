使用whisper完成中文语音对话，并调用大模型回复。

```bash
conda create --name whisper_env python=3.9

cd D:\whisper_project\
conda activate whisper_env
ffmpeg -version

conda install -c conda-forge ffmpeg

pip install PyAudio
pip install openai-whisper 
pip install gradio==3.41.2

pip install zhconv
pip install langchain_openai
pip install langchain
```

whisper_project