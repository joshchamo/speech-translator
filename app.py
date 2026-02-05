import gradio as gr
import requests
import os
import base64

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

# Mapping for Translation (mBART-50) and TTS (MMS)
# Note: MMS TTS uses 3-letter ISO codes (fra, spa, deu)
LANG_DATA = {
    "English": {"mbart": "en_XX", "mms": "eng"},
    "Spanish": {"mbart": "es_XX", "mms": "spa"},
    "French": {"mbart": "fr_XX", "mms": "fra"},
    "German": {"mbart": "de_DE", "mms": "deu"},
    "Hindi": {"mbart": "hi_IN", "mms": "hin"},
    "Japanese": {"mbart": "ja_XX", "mms": "jpn"},
    "Russian": {"mbart": "ru_RU", "mms": "rus"},
    "Turkish": {"mbart": "tr_TR", "mms": "tur"},
    "Vietnamese": {"mbart": "vi_VN", "mms": "vie"}
}

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(url, payload, is_audio_in=False, is_audio_out=False):
    h = headers.copy()
    if is_audio_in:
        h["Content-Type"] = "audio/wav"
        with open(payload, "rb") as f: data = f.read()
        response = requests.post(url, headers=h, data=data)
    else:
        response = requests.post(url, headers=h, json=payload)
    
    if response.status_code != 200:
        return None
    
    return response.content if is_audio_out else response.json()

def text_to_speech(text, language):
    if not text: return None
    mms_code = LANG_DATA[language]["mms"]
    tts_url = f"https://router.huggingface.co/hf-inference/models/facebook/mms-tts-{mms_code}"
    
    audio_content = query_api(tts_url, {"inputs": text}, is_audio_out=True)
    if audio_content:
        # Save to a temporary file for Gradio to play
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(audio_content)
        return out_path
    return None

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None
    
    # 1. Transcribe
    asr_res = query_api(WHISPER_URL, audio_path, is_audio_in=True)
    transcript = asr_res.get("text", "Error") if asr_res else "ASR Failed"

    # 2. Translate
    payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_DATA[input_lang]["mbart"],
            "tgt_lang": LANG_DATA[target_lang]["mbart"]
        }
    }
    tr_res = query_api(TRANSLATE_URL, payload)
    translation = tr_res[0].get("translation_text", "Error") if tr_res else "Translation Failed"
    
    # 3. Generate Speech Automatically (Optional)
    audio_out = text_to_speech(translation, target_lang)
    
    return transcript, translation, audio_out

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 VOXTRAL: Voice-to-Voice Translator")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Speak")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Translated Voice", interactive=False)
    
    # Trigger all three steps at once
    audio_in.stop_recording(
        translate_speech, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )
    
    # Manual "Speak Again" Button
    speak_btn = gr.Button("🔊 Read Translation Again")
    speak_btn.click(text_to_speech, inputs=[trn_out, out_lang], outputs=audio_out)

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)