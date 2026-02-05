import gradio as gr
import requests
import os
import time

# Use the 2026 Hugging Face Router endpoints
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/m2m100_418M"
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# 10 Popular languages with M2M100 codes
LANG_CODES = {
    "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh", 
    "Japanese": "ja", "Hindi": "hi", "Arabic": "ar", "Portuguese": "pt",
    "Russian": "ru", "Italian": "it"
}

def fast_api_call(url, payload, is_audio=False):
    h = headers.copy()
    if is_audio:
        h["Content-Type"] = "audio/wav"
        with open(payload, "rb") as f:
            data = f.read()
        return requests.post(url, headers=h, data=data)
    return requests.post(url, headers=h, json=payload)

def process_voice(audio_path, target_lang):
    if not audio_path: return "No audio.", ""
    
    # 1. Faster Transcription
    res_asr = fast_api_call(WHISPER_URL, audio_path, is_audio=True).json()
    transcript = res_asr.get("text", "Error in ASR")
    
    # 2. Faster Translation (M2M100)
    # This model uses ISO codes (es, fr, zh) rather than the long NLLB codes
    payload = {
        "inputs": transcript,
        "parameters": {"forced_bos_token_id": f"[{LANG_CODES[target_lang]}]"}
    }
    
    res_tr = fast_api_call(TRANSLATE_URL, payload).json()
    
    # Handle the list response format
    try:
        translation = res_tr[0].get("translation_text", "Error")
    except:
        translation = "Model warming up... try again in 5s."
        
    return transcript, translation

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ High-Speed Multi-Translator")
    
    with gr.Row():
        audio_input = gr.Audio(sources="microphone", type="filepath", label="Speak")
        lang_input = gr.Dropdown(choices=list(LANG_CODES.keys()), value="Spanish", label="Target Language")
    
    with gr.Row():
        text_trans = gr.Textbox(label="Transcript (English)")
        text_final = gr.Textbox(label="Translated")

    # This trigger starts the process immediately after recording
    audio_input.stop_recording(process_voice, [audio_input, lang_input], [text_trans, text_final])

    # PRO TIP: Warm up both models when the page loads
    demo.load(lambda: [
        requests.post(WHISPER_URL, headers=headers, data=b""),
        requests.post(TRANSLATE_URL, headers=headers, json={"inputs": "init"})
    ])

demo.launch(server_name="0.0.0.0", server_port=7860)