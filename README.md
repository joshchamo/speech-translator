---
title: Voxtral Multimodal Translator
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
short_description: 50-Language Speech Translator with Whisper & mBART
models:
- openai/whisper-large-v3-turbo
- facebook/mbart-large-50-many-to-many-mmt
tags:
- speech-to-speech
- translation
- automatic-speech-recognition
- audio-to-audio
- multilingual
- whisper
- mbart
---

# 🌍 Voxtral: Universal Speech-to-Speech Translator

Voxtral is a high-performance, multimodal AI demonstration that transcribes, translates, and voices speech in real-time. By chaining three state-of-the-art technologies, it creates a seamless "Universal Translator" experience across 50 languages.

## 🚀 The AI Pipeline
To ensure high accuracy and low latency, Voxtral utilizes a modular inference chain:

1. **Transcription (ASR):** [OpenAI Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) 
   - Converts raw microphone input into high-fidelity text.
2. **Translation (NMT):** [Facebook mBART-50 Many-to-Many](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) 
   - Translates the transcription into the target language while preserving context across 50+ language pairs.
3. **Speech Synthesis (TTS):** Powered by `gTTS` 
   - Generates a natural-sounding voice output in the target language for immediate playback.

## 📖 Key Features
- **50 Languages Supported:** From Arabic and Chinese to Vietnamese and Xhosa.
- **Multimodal Workflow:** Processes audio input and provides both text and audio output.
- **Serverless Inference:** Powered by Hugging Face's hosted inference API for rapid processing.

## 🛠️ How to Use
1. Select your **Input Language** and **Target Language**.
2. Click the **Microphone** icon and speak a sentence.
3. Once you stop recording, the system will automatically transcribe, translate, and play the spoken result.
4. Use the **Debug Logs** accordion if you want to see the system's status and API response times.

## 🧪 Technical Details
This Space demonstrates the power of **Inference Chaining**. Instead of relying on a single massive model, 
it uses specialized models for each task, allowing for a more flexible and robust translation experience. 
The backend is built with Python and Gradio, communicating via the Hugging Face Inference API.