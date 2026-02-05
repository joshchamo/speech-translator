import gradio as gr
import requests
import os
import sys

# Ensure logs show up immediately
sys.stdout.reconfigure(line_buffering=True)

# We'll use direct requests to the API for maximum stability
API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_api(url, data=None, is_audio=False):
    if is_audio:
        with open(data, "rb") as f:
            response = requests.post(url, headers=headers, data=f)
    else:
        response = requests.post(url, headers=headers, json={"inputs": data})
    return response.json()

def process_all(audio_path):
    if not audio_path:
        return "No audio", ""

    try:
        # 1. Transcribe
        print("Starting Transcription...")
        asr_json = query_api(WHISPER_URL, data=audio_path, is_audio=True)
        # Handle cases where API returns an error message instead of text
        transcript = asr_json.get("text", str(asr_json))
        print(f"Transcript: {transcript}")

        # 2. Translate
        print("Starting Translation...")
        trans_json = query_api(TRANSLATE_URL, data=transcript)
        # Standard Helsinki-NLP return format is a list of dicts
        if isinstance(trans_json, list) and len(trans_json) > 0:
            translation = trans_json[0].get("translation_text", "Translation Error")
        else:
            translation = f"API error: {trans_json}"
        
        return transcript, translation

    except Exception as e:
        return f"System Error: {str(e)}", ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Speech-to-French Translator")
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath")
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="French Translation")
    
    audio_in.stop_recording(process_all, inputs=audio_in, outputs=[txt_out, trn_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)