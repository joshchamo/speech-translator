---
title: Voxtral Universal S2S
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: A 50-language Speech-to-Speech translation pipeline.
---

# 🌍 Voxtral: Universal Speech-to-Speech Translator

Voxtral is a multimodal AI demo that translates spoken language into another language's voice in real-time. It chains together three distinct AI technologies to achieve a seamless "Universal Translator" experience.

## 🚀 How it Works
The application follows a linear pipeline to process your voice:

1. **Automatic Speech Recognition (ASR):** Powered by `OpenAI Whisper Large v3 Turbo`, it transcribes your spoken input into text with high accuracy.
2. **Neural Machine Translation (NMT):** Powered by `Facebook mBART-50`, the transcribed text is translated into one of 50 supported languages.
3. **Text-to-Speech (TTS):** Powered by `gTTS`, the translated text is converted back into a natural-sounding voice in the target language.



## 🛠️ Tech Stack
- **Frontend:** [Gradio](https://gradio.app/)
- **Inference:** [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- **Audio Engine:** gTTS (Google Text-to-Speech)

## 📖 Supported Languages
This demo supports 50 languages, including Arabic, Chinese, English, French, German, Hindi, Japanese, Spanish, Vietnamese, and many more.
