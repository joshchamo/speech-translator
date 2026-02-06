import gradio as gr
import os
import json
import sys
import requests
from huggingface_hub import InferenceClient

# Ensure real-time logging
sys.stdout.reconfigure(line_buffering=True)

print("--- INITIALIZING VOXTRAL V9 ---", flush=True)

API_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=API_TOKEN)

# THE NEW 2026 ROUTER ENDPOINT
# Old: https://api-inference.huggingface.co/models/
# New: https://router.huggingface.co/hf-inference/models/
ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

LANG_DATA = {
    "English": "en_XX", 
    "Spanish": "es_XX", 
    "French": "fr_XX", 
    "German": "de_DE", 
    "Japanese": "ja_XX"
}

def translate_and_speak(audio_path, in_lang, out_lang):
    if not audio_path: return "", "", None
    
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        # 1. Transcription (Whisper-v3)
        print("DEBUG: Transcribing speech...", flush=True)
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text
        print(f"DEBUG: Found text: {transcript}", flush=True)

        # 2. Translation (Using the NEW Router URL)
        print("DEBUG: Requesting translation from Router...", flush=True)
        translate_url = f"{ROUTER_BASE}/{TRANSLATE_MODEL}"
        
        payload = {
            "inputs": transcript,
            "parameters": {
                "src_lang": LANG_DATA[in_lang], 
                "tgt_lang": LANG_DATA[out_lang]
            }
        }
        
        response = requests.post(translate_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Router Error {response.status_code}: {response.text}")
            
        translation = response.json()[0]['translation_text']
        print(f"DEBUG: Translated to: {translation}", flush=True)

        # 3. Text-to-Speech (MMS)
        lang_short = LANG_DATA[out_lang][:2]
        tts_model = f"facebook/mms-tts-{lang_short}"
        print(f"DEBUG: Generating audio via {tts_model}...", flush=True)
        
        audio_content = client.text_to_speech(translation, model=tts_model)
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(audio_content)
            
        return transcript, translation, out_path
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        return f"System Alert: {str(e)}", "Please check router status", None

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL v9: Router-Powered")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="Translate From")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="Translate To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Voice Output", autoplay=True)
    
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft()
    )