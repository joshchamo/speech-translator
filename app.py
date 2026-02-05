import gradio as gr
import requests
import os

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"
# Switching to a more modern, pinned multilingual TTS model
TTS_URL = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Update data mapping to match Qwen3's supported languages
LANG_DATA = {
    "English": {"mbart": "en_XX", "qwen": "English"},
    "Spanish": {"mbart": "es_XX", "qwen": "Spanish"},
    "French": {"mbart": "fr_XX", "qwen": "French"},
    "German": {"mbart": "de_DE", "qwen": "German"},
    "Japanese": {"mbart": "ja_XX", "qwen": "Japanese"},
    "Chinese": {"mbart": "zh_CN", "qwen": "Chinese"}
}

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def text_to_speech(text, language):
    if not text: return None
    
    # Qwen3 uses natural language names or ISO codes in the prompt
    payload = {
        "inputs": text,
        "parameters": {"language": LANG_DATA[language]["qwen"]}
    }
    
    print(f"--- Calling TTS for {language} ---")
    response = requests.post(TTS_URL, headers=headers, json=payload)
    
    if response.status_code == 200 and "audio" in response.headers.get("Content-Type", ""):
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(response.content)
        return out_path
    
    # Log the exact error for debugging
    print(f"TTS Error {response.status_code}: {response.text[:100]}")
    return None

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None
    
    # 1. Transcribe
    with open(audio_path, "rb") as f:
        asr_res = requests.post(WHISPER_URL, headers=headers, data=f.read())
    transcript = asr_res.json().get("text", "Error")

    # 2. Translate
    tr_payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_DATA[input_lang]["mbart"],
            "tgt_lang": LANG_DATA[target_lang]["mbart"]
        }
    }
    tr_res = requests.post(TRANSLATE_URL, headers=headers, json=tr_payload)
    translation = tr_res.json()[0].get("translation_text", "")

    # 3. TTS
    audio_out = text_to_speech(translation, target_lang)
    return transcript, translation, audio_out

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 Universal Voice Translator (v3)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Input")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Playback", autoplay=True)
    speak_btn = gr.Button("🔊 Speak Translation")

    audio_in.stop_recording(translate_speech, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])
    speak_btn.click(text_to_speech, [trn_out, out_lang], audio_out)

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)