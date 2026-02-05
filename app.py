import gradio as gr
import requests
import os
import sys

# Immediate log flushing for HF Container tab
sys.stdout.reconfigure(line_buffering=True)

API_TOKEN = os.getenv("HF_TOKEN")
# Using the most stable 'pinned' models on the HF Router
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
# Switching to a more stable translation path for the demo
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-fr"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(url, payload, is_audio=False):
    current_headers = headers.copy()
    if is_audio:
        current_headers["Content-Type"] = "audio/wav"
        with open(payload, "rb") as f:
            data = f.read()
        response = requests.post(url, headers=current_headers, data=data)
    else:
        response = requests.post(url, headers=current_headers, json=payload)
    
    # Return status and raw text if JSON fails to prevent the crash you saw
    try:
        return response.status_code, response.json()
    except:
        return response.status_code, {"error": "Non-JSON response", "details": response.text[:200]}

def translate_speech(audio_path, target_lang):
    if not audio_path:
        return "No audio recorded.", "", "Empty input."

    log_entry = f"--- New Request ---\n"
    
    # 1. Transcription
    status_asr, asr_data = query_api(WHISPER_URL, audio_path, is_audio=True)
    if status_asr != 200:
        err = asr_data.get('error', 'Unknown ASR Error')
        return f"Error: {err}", "", f"ASR Status {status_asr}: {asr_data}"
    
    transcript = asr_data.get("text", "")
    log_entry += f"Transcript: {transcript}\n"

    # 2. Translation
    # Note: Helsinki models are pair-specific. For a multi-lang demo,
    # NLLB is better, but if it gives 404, we check for it here:
    status_tr, tr_data = query_api(TRANSLATE_URL, {"inputs": transcript})
    
    if status_tr != 200:
        return transcript, "Translation Model Offline (404/503)", f"TR Status {status_tr}: {tr_data}"

    # Extract translation from list format
    try:
        translation = tr_data[0].get("translation_text", "Result missing")
    except:
        translation = "Parsing Error"

    return transcript, translation, log_entry + "Success!"

# --- UI Definition ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL Mini-Demo")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record Speech")
        # For now, let's keep it to French to ensure stability
        lang_choice = gr.Dropdown(choices=["French"], value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="English Transcription")
        trn_out = gr.Textbox(label="Translation")

    with gr.Accordion("System Logs", open=False):
        debug_out = gr.Textbox(label="Raw API Trace")

    audio_in.stop_recording(
        translate_speech, 
        inputs=[audio_in, lang_choice], 
        outputs=[txt_out, trn_out, debug_out]
    )

if __name__ == "__main__":
    # Fix for Gradio 6.0: Move theme and ssr_mode here
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(),
        ssr_mode=False
    )