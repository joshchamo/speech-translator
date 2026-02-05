import gradio as gr
import requests
import os
import sys

# Immediate log flushing
sys.stdout.reconfigure(line_buffering=True)

API_TOKEN = os.getenv("HF_TOKEN")
# Using the new 2026 Router addresses
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
# Switching to a more stable multilingual model for the demo
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Map for MBART (uses language_COUNTRY codes)
LANG_CODES = {
    "Spanish": "es_XX", "French": "fr_XX", "German": "de_DE", 
    "Chinese": "zh_CN", "Arabic": "ar_AR", "Hindi": "hi_IN"
}

def query_api(url, payload, is_audio=False):
    h = headers.copy()
    try:
        if is_audio:
            # FIX: Explicitly set the audio type
            h["Content-Type"] = "audio/wav"
            with open(payload, "rb") as f:
                data = f.read()
            response = requests.post(url, headers=h, data=data, timeout=15)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=15)
        
        # Check if it's JSON before parsing
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}", "details": response.text[:100]}
    except Exception as e:
        return {"error": "Connection Failed", "details": str(e)}

def fast_process(audio_path, target_lang):
    if not audio_path: return "No audio.", ""
    
    # 1. Transcribe
    asr_data = query_api(WHISPER_URL, audio_path, is_audio=True)
    if "error" in asr_data:
        return f"ASR Error: {asr_data['error']}", "Check logs"
    
    transcript = asr_data.get("text", "")
    
    # 2. Translate
    target_code = LANG_CODES[target_lang]
    payload = {
        "inputs": transcript,
        "parameters": {"src_lang": "en_XX", "tgt_lang": target_code}
    }
    
    tr_data = query_api(TRANSLATE_URL, payload)
    
    try:
        # MBART usually returns a list of dicts
        translation = tr_data[0].get("translation_text", "Result missing")
    except:
        translation = f"Trans Error: {tr_data.get('error', 'Unknown')}"
        
    return transcript, translation

with gr.Blocks() as demo:
    gr.Markdown("# ⚡ Voxtral Ultra-Fast Demo")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Speak")
        lang_in = gr.Dropdown(choices=list(LANG_CODES.keys()), value="French", label="To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")

    audio_in.stop_recording(fast_process, [audio_in, lang_in], [txt_out, trn_out])

# Fix for Gradio 6.0 warnings: Move theme to launch()
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(),
        ssr_mode=False
    )