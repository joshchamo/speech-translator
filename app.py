import gradio as gr
import os
import json
import sys
import requests
from huggingface_hub import InferenceClient

sys.stdout.reconfigure(line_buffering=True)

print("--- INITIALIZING VOXTRAL V12 ---", flush=True)

API_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=API_TOKEN)
ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

# Mapping back to specific repos which are more likely to be cached/pinned
LANG_DATA = {
    "English": {"mbart": "en_XX", "tts": "facebook/mms-tts-eng"}, 
    "Spanish": {"mbart": "es_XX", "tts": "facebook/mms-tts-spa"}, 
    "French": {"mbart": "fr_XX", "tts": "facebook/mms-tts-fra"}, 
    "German": {"mbart": "de_DE", "tts": "facebook/mms-tts-deu"}, 
    "Japanese": {"mbart": "ja_XX", "tts": "facebook/mms-tts-jpn"}
}

def translate_and_speak(audio_path, in_lang, out_lang):
    if not audio_path: return "", "", None
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        # 1. Transcription
        print("DEBUG: STT...", flush=True)
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation
        print("DEBUG: Translation...", flush=True)
        tr_url = f"{ROUTER_BASE}/{TRANSLATE_MODEL}"
        tr_payload = {
            "inputs": transcript,
            "parameters": {"src_lang": LANG_DATA[in_lang]["mbart"], "tgt_lang": LANG_DATA[out_lang]["mbart"]}
        }
        tr_resp = requests.post(tr_url, headers=headers, json=tr_payload, timeout=30)
        translation = tr_resp.json()[0]['translation_text']

        # 3. TTS with Fallback Logic
        target_tts_model = LANG_DATA[out_lang]["tts"]
        print(f"DEBUG: Attempting TTS with {target_tts_model}...", flush=True)
        
        tts_url = f"{ROUTER_BASE}/{target_tts_model}"
        tts_resp = requests.post(tts_url, headers=headers, json={"inputs": translation}, timeout=30)
        
        # If the specific language is 404 (not pinned), fallback to a high-availability English model
        # so the app doesn't just "die"
        if tts_resp.status_code == 404:
            print(f"WARN: {target_tts_model} not found on Router. Falling back to English TTS.", flush=True)
            fallback_url = f"{ROUTER_BASE}/facebook/mms-tts-eng"
            tts_resp = requests.post(fallback_url, headers=headers, json={"inputs": translation}, timeout=30)

        if tts_resp.status_code != 200:
            raise Exception(f"TTS Error {tts_resp.status_code}: {tts_resp.text}")
            
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(tts_resp.content)
            
        return transcript, translation, out_path
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        return f"Error: {str(e)}", "Check Logs", None

# --- UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL v12 (Smart Fallback)")
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    txt_out = gr.Textbox(label="Transcription")
    trn_out = gr.Textbox(label="Translation")
    audio_out = gr.Audio(label="Spoken Result", autoplay=True)
    
    audio_in.stop_recording(translate_and_speak, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())