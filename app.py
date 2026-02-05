import gradio as gr
import requests
import os

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

# Mapping for Translation and TTS
LANG_DATA = {
    "English": {"mbart": "en_XX", "mms": "eng"},
    "Spanish": {"mbart": "es_XX", "mms": "spa"},
    "French": {"mbart": "fr_XX", "mms": "fra"},
    "German": {"mbart": "de_DE", "mms": "deu"},
    "Hindi": {"mbart": "hi_IN", "mms": "hin"},
    "Japanese": {"mbart": "ja_XX", "mms": "jpn"}
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
    
    if response.status_code != 200: return None
    return response.content if is_audio_out else response.json()

def text_to_speech(text, language):
    if not text or language not in LANG_DATA: return None
    mms_code = LANG_DATA[language]["mms"]
    tts_url = f"https://router.huggingface.co/hf-inference/models/facebook/mms-tts-{mms_code}"
    
    audio_content = query_api(tts_url, {"inputs": text}, is_audio_out=True)
    if audio_content:
        out_path = "output.wav"
        with open(out_path, "wb") as f: f.write(audio_content)
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
    translation = tr_res[0].get("translation_text", "") if tr_res else ""
    
    # 3. Generate initial speech
    audio_out = text_to_speech(translation, target_lang)
    return transcript, translation, audio_out

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 VOXTRAL Voice-to-Voice")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="1. Speak Here")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription (Source)")
        with gr.Column():
            trn_out = gr.Textbox(label="Translation (Target)")
            # THE NEW BUTTON:
            speak_btn = gr.Button("🔊 Speak Translation", variant="primary")
    
    # The output audio player (hidden or visible)
    # autoplay=True ensures it plays as soon as the translation is done
    audio_out = gr.Audio(label="Audio Output", autoplay=True, interactive=False)
    
    # Event 1: Automatic processing when recording stops
    audio_in.stop_recording(
        translate_speech, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )
    
    # Event 2: Manual trigger when button is clicked
    speak_btn.click(
        text_to_speech, 
        inputs=[trn_out, out_lang], 
        outputs=audio_out
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)