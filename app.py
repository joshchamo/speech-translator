import gradio as gr
import requests
import os
import time

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
NLLB_URL = "https://router.huggingface.co/hf-inference/models/facebook/nllb-200-distilled-600M"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Mapping user-friendly names to NLLB language codes
LANG_MAP = {
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva",
    "Russian": "rus_Cyrl",
    "Turkish": "tur_Latn",
    "Dutch": "nld_Latn"
}

def query_api(url, payload, is_audio=False):
    current_headers = headers.copy()
    if is_audio:
        current_headers["Content-Type"] = "audio/wav"
        with open(payload, "rb") as f:
            data = f.read()
        response = requests.post(url, headers=current_headers, data=data)
    else:
        response = requests.post(url, headers=current_headers, json=payload)
    
    return response.status_code, response.json()

def translate_speech(audio_path, target_lang_name):
    if not audio_path: return "No audio recorded.", ""
    
    # 1. Transcription
    status_asr, asr_data = query_api(WHISPER_URL, audio_path, is_audio=True)
    if status_asr != 200: return f"ASR Error: {asr_data}", ""
    transcript = asr_data.get("text", "")

    # 2. Translation using NLLB
    target_code = LANG_MAP[target_lang_name]
    payload = {
        "inputs": transcript,
        "parameters": {"src_lang": "eng_Latn", "tgt_lang": target_code}
    }
    
    status_tr, tr_data = query_api(NLLB_URL, payload)
    
    # Handle the "Model Loading" 503 error gracefully
    if status_tr == 503:
        return transcript, "Model is waking up... please try again in 10 seconds."
    
    # NLLB usually returns a list of dictionaries
    try:
        translation = tr_data[0].get("translation_text", "Error")
    except:
        translation = f"API Error: {tr_data}"
        
    return transcript, translation

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Multilingual Speech Translator")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="1. Record English Speech")
        lang_dropdown = gr.Dropdown(
            choices=list(LANG_MAP.keys()), 
            value="French", 
            label="2. Select Target Language"
        )
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription (English)")
        trn_out = gr.Textbox(label="Translation")

    # Trigger translation when recording stops
    audio_in.stop_recording(
        translate_speech, 
        inputs=[audio_in, lang_dropdown], 
        outputs=[txt_out, trn_out]
    )

    # Background Warm-up
    demo.load(lambda: requests.post(NLLB_URL, headers=headers, json={"inputs": "warmup"}))

demo.launch(server_name="0.0.0.0", server_port=7860)