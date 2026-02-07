import gradio as gr
import os
import json
import sys
import requests
from huggingface_hub import InferenceClient

# Ensure real-time logging
sys.stdout.reconfigure(line_buffering=True)

print("--- INITIALIZING VOXTRAL V11 ---", flush=True)

API_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=API_TOKEN)

# Unified Router Endpoint
ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
TTS_MODEL = "facebook/mms-tts"

# Updated Mapping
LANG_DATA = {
    "English": {"mbart": "en_XX", "mms": "eng"}, 
    "Spanish": {"mbart": "es_XX", "mms": "spa"}, 
    "French": {"mbart": "fr_XX", "mms": "fra"}, 
    "German": {"mbart": "de_DE", "mms": "deu"}, 
    "Japanese": {"mbart": "ja_XX", "mms": "jpn"}
}

def translate_and_speak(audio_path, in_lang, out_lang):
    if not audio_path: return "", "", None
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        # 1. Transcription
        print("DEBUG: Calling STT...", flush=True)
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation
        print("DEBUG: Calling Translation...", flush=True)
        translate_url = f"{ROUTER_BASE}/{TRANSLATE_MODEL}"
        tr_payload = {
            "inputs": transcript,
            "parameters": {
                "src_lang": LANG_DATA[in_lang]["mbart"], 
                "tgt_lang": LANG_DATA[out_lang]["mbart"]
            }
        }
        tr_resp = requests.post(translate_url, headers=headers, json=tr_payload, timeout=30)
        translation = tr_resp.json()[0]['translation_text']

        # 3. TTS (Using raw request to pass 'language' parameter)
        print(f"DEBUG: Calling TTS for {out_lang}...", flush=True)
        tts_url = f"{ROUTER_BASE}/{TTS_MODEL}"
        mms_code = LANG_DATA[out_lang]["mms"]
        
        tts_payload = {
            "inputs": translation,
            "parameters": {"language": mms_code}
        }
        
        tts_resp = requests.post(tts_url, headers=headers, json=tts_payload, timeout=30)
        
        if tts_resp.status_code != 200:
            raise Exception(f"TTS Router Error {tts_resp.status_code}: {tts_resp.text}")
            
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(tts_resp.content)
            
        return transcript, translation, out_path
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        return f"Error: {str(e)}", "Please check logs", None

# --- UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL v11 (Final Stability)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    txt_out = gr.Textbox(label="Transcript")
    trn_out = gr.Textbox(label="Translation")
    audio_out = gr.Audio(label="Voice Result", autoplay=True)
    
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())