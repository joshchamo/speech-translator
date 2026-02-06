import gradio as gr
import os
import json
import sys
import requests
from huggingface_hub import InferenceClient

# Unbuffered logging for real-time debugging
sys.stdout.reconfigure(line_buffering=True)

print("--- INITIALIZING VOXTRAL V10 ---", flush=True)

API_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=API_TOKEN)

# The correct 2026 router path
ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
# THE NEW UNIFIED REPO (Replacing the 1100+ separate repos)
TTS_MODEL_BASE = "facebook/mms-tts"

# ISO 639-3 codes for MMS
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
        print("DEBUG: STT start...", flush=True)
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation
        print("DEBUG: Translation start...", flush=True)
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

        # 3. TTS with Language Adapter (The fix for your 404)
        print(f"DEBUG: TTS start for {out_lang}...", flush=True)
        mms_code = LANG_DATA[out_lang]["mms"]
        
        # We call the BASE model and pass the language as a parameter
        # This is how the Inference API handles the 1100+ MMS languages now
        audio_content = client.text_to_speech(
            translation, 
            model=TTS_MODEL_BASE,
            parameters={"language": mms_code}
        )
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(audio_content)
            
        return transcript, translation, out_path
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        return f"System Error: {str(e)}", "Please check language codes", None

# --- UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL v10: Unified TTS Repo")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Speak")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    txt_out = gr.Textbox(label="Transcript")
    trn_out = gr.Textbox(label="Translation")
    audio_out = gr.Audio(label="Spoken Result", autoplay=True)
    
    audio_in.stop_recording(translate_and_speak, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())