import gradio as gr
import requests
import os

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Expanded Mapping (mBART-50 compatible)
LANG_CODES = {
    "English": "en_XX", "Spanish": "es_XX", "French": "fr_XX", 
    "German": "de_DE", "Chinese": "zh_CN", "Arabic": "ar_AR", 
    "Hindi": "hi_IN", "Japanese": "ja_XX", "Russian": "ru_RU"
}

def query_api(url, payload, is_audio=False):
    h = headers.copy()
    if is_audio:
        h["Content-Type"] = "audio/wav"
        with open(payload, "rb") as f:
            data = f.read()
        response = requests.post(url, headers=h, data=data)
    else:
        response = requests.post(url, headers=h, json=payload)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Error {response.status_code}", "details": response.text}

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "No audio.", ""
    
    # 1. Transcribe (Whisper detects lang or uses provided hint)
    # Hint: We can pass 'language' to Whisper to improve accuracy
    asr_payload = audio_path
    res_asr = query_api(WHISPER_URL, as_asr_payload, is_audio=True)
    transcript = res_asr.get("text", "Transcription failed.")

    # 2. Translate (Source to Target)
    src_code = LANG_CODES[input_lang]
    tgt_code = LANG_CODES[target_lang]
    
    tr_payload = {
        "inputs": transcript,
        "parameters": {"src_lang": src_code, "tgt_lang": tgt_code}
    }
    
    res_tr = query_api(TRANSLATE_URL, tr_payload)
    
    try:
        translation = res_tr[0].get("translation_text", "Translation missing.")
    except:
        translation = "Model is busy or loading. Please try again."
        
    return transcript, translation

with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Universal Voice Translator")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Speak")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_CODES.keys()), value="English", label="Input Language")
            out_lang = gr.Dropdown(choices=list(LANG_CODES.keys()), value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Original Text")
        trn_out = gr.Textbox(label="Translated Text")

    audio_in.stop_recording(translate_speech, [audio_in, in_lang, out_lang], [txt_out, trn_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), ssr_mode=False)