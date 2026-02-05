import gradio as gr
import requests
import os

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"
# Switching to Kokoro: The most stable serverless TTS model in 2026
TTS_URL = "https://router.huggingface.co/hf-inference/models/hexgrad/Kokoro-82M"

LANG_DATA = {
    "English": {"mbart": "en_XX", "kokoro": "en-us"},
    "Spanish": {"mbart": "es_XX", "kokoro": "es"},
    "French": {"mbart": "fr_XX", "kokoro": "fr-fr"},
    "Japanese": {"mbart": "ja_XX", "kokoro": "ja"},
    "Chinese": {"mbart": "zh_CN", "kokoro": "zh"}
}

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def text_to_speech(text, language):
    if not text: return None
    
    # Kokoro uses the 'voice' parameter to determine language
    payload = {
        "inputs": text,
        "parameters": {"voice": LANG_DATA[language]["kokoro"]}
    }
    
    print(f"--- Requesting Kokoro TTS for {language} ---")
    response = requests.post(TTS_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(response.content)
        return out_path
    
    print(f"TTS Failed ({response.status_code}): {response.text[:100]}")
    return None

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None
    
    # 1. Transcribe
    with open(audio_path, "rb") as f:
        asr_res = requests.post(WHISPER_URL, headers=headers, data=f.read())
    transcript = asr_res.json().get("text", "Error")

    # 2. Translate
    tr_payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_DATA[input_lang]["mbart"],
            "tgt_lang": LANG_DATA[target_lang]["mbart"]
        }
    }
    tr_res = requests.post(TRANSLATE_URL, headers=headers, json=tr_payload)
    translation = tr_res.json()[0].get("translation_text", "")

    # 3. TTS
    audio_out = text_to_speech(translation, target_lang)
    return transcript, translation, audio_out

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 VOXTRAL v3: High-Reliability Translator")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="Speak In")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="Translate To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Original")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Voice Output", autoplay=True)
    speak_btn = gr.Button("🔊 Read Again", variant="secondary")

    audio_in.stop_recording(translate_speech, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])
    speak_btn.click(text_to_speech, [trn_out, out_lang], audio_out)

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)