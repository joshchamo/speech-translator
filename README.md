---
title: 50-Language Speech-to-Speech Translator using Whisper & mBART
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: 50-Language Speech Translator with Whisper & mBART.
models:
- openai/whisper-large-v3-turbo
- facebook/mbart-large-50-many-to-many-mmt
- engine gTTS (Google Text-to-Speech)
tags:
  - speech-to-text
  - speech-translation
  - speech-to-speech
  - transcription
  - translation
  - automatic-speech-recognition
  - text-to-speech
  - audio-to-audio
  - multilingual
  - whisper
  - mbart
  - gtts
  - gradio
  - audio
  - voice
sdk_version: 6.5.1
---

# 🌍 50-Language Speech-to-Speech Translator

This Hugging Face Space is a **multimodal demo** that performs end-to-end speech translation by chaining together speech recognition, machine translation, and text-to-speech synthesis.

It allows users to speak in one language and hear the translated speech in another, supporting 50 languages.

---

## 🚀 How It Works

The application follows a linear processing pipeline:

1. **Automatic Speech Recognition (ASR)**  
   Spoken audio is transcribed into text using **Whisper (Large v3 Turbo)**.

2. **Neural Machine Translation (NMT)**  
   The transcribed text is translated into a selected target language using **mBART-50**, which supports 50 languages.

3. **Text-to-Speech (TTS)**  
   The translated text is converted back into audio using **gTTS (Google Text-to-Speech)**.

The result is a seamless speech-to-speech translation experience.

---

## 🛠️ Tech Stack

- **UI:** Gradio  
- **Speech Recognition:** Whisper  
- **Translation:** Facebook mBART-50  
- **Text-to-Speech:** gTTS  
- **Hosting:** Hugging Face Spaces  

---

## 🌐 Supported Languages

The demo supports 50 languages, including but not limited to:

Arabic, Chinese, English, French, German, Hindi, Japanese, Korean, Portuguese, Spanish, Vietnamese, and many more.

---

## ⚙️ Design Notes

- Models were selected to balance **language coverage**, **latency**, and **availability** on Hugging Face Spaces.
- Always-on or fast-loading models were preferred to avoid cold-start delays.
- The demo focuses on clarity and reliability rather than pushing the largest possible models.

---

## 📌 Limitations

- Long audio inputs may increase processing time.
- Translation quality can vary for less common language pairs.
- TTS voices depend on gTTS language support.

---

## 📄 License

This project is released under the **MIT License**.