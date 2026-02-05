import gradio as gr
import requests
import os
import sys
import json

# Ensure logs show up immediately
sys.stdout.reconfigure(line_buffering=True)

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-fr"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(url, data=None, is_audio=False):
    if is_audio:
        with open(data, "rb") as f:
            response = requests.post(url, headers=headers, data=f)
    else:
        response = requests.post(url, headers=headers, json={"inputs": data})
    
    # 1. Check if the response is valid JSON
    try:
        return response.status_code, response.json()
    except Exception:
        # If not JSON, return the raw text (likely an HTML error)
        return response.status_code, response.text[:500] 

def process_all(audio_path, current_logs):
    if not audio_path:
        return "No audio", "", current_logs

    new_log = "--- New Request ---\n"
    try:
        # 1. Transcribe
        status_asr, asr_data = query_api(WHISPER_URL, data=audio_path, is_audio=True)
        new_log += f"ASR Status: {status_asr}\n"
        
        if status_asr != 200:
            new_log += f"ASR Error Body: {asr_data}\n"
            return f"ASR Error ({status_asr})", "", current_logs + "\n" + new_log
            
        transcript = asr_data.get("text", "No text found")
        
        # 2. Translate
        status_tr, tr_data = query_api(TRANSLATE_URL, data=transcript)
        new_log += f"Trans Status: {status_tr}\n"
        
        if status_tr != 200:
            new_log += f"Trans Error Body: {tr_data}\n"
            return transcript, f"Translation Error ({status_tr})", current_logs + "\n" + new_log
            
        translation = tr_data[0].get("translation_text", "Error")
        return transcript, translation, current_logs + "\n" + new_log + "Success!"

    except Exception as e:
        return "System Error", "", current_logs + f"\nException: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Speech-to-French Prototype")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="French Translation")

    # Debugging Section (Collapsible)
    with gr.Accordion("🛠️ Debug Logs", open=False):
        log_box = gr.Textbox(label="Raw API Responses", lines=10, interactive=False)
        clear_btn = gr.Button("Clear Logs")

    # State to keep logs persistent
    history = gr.State("")
    
    audio_in.stop_recording(
        process_all, 
        inputs=[audio_in, history], 
        outputs=[txt_out, trn_out, log_box]
    ).then(lambda x: x, inputs=log_box, outputs=history) # Update history state

    clear_btn.click(lambda: ("", ""), outputs=[log_box, history])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)