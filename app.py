import gradio as gr
import requests
import os

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

LANG_DATA = {
    "English": {"mbart": "en_XX", "mms": "eng"},
    "Spanish": {"mbart": "es_XX", "mms": "spa"},
    "French": {"mbart": "fr_XX", "mms": "fra"},
    "German": {"mbart": "de_DE", "mms": "deu"},
    "Hindi": {"mbart": "hi_IN", "mms": "hin"},
    "Japanese": {"mbart": "ja_XX", "mms": "jpn"}
}

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def text_to_speech(text, language):
    if not text: return None
    mms_code = LANG_DATA[language]["mms"]
    tts_url = f"https://router.huggingface.co/hf-inference/models/facebook/mms-tts-{mms_code}"
    
    print(f"--- TTS Request for {language} ---")
    response = requests.post(tts_url, headers=headers, json={"inputs": text})
    
    # Check if the response is actually audio
    content_type = response.headers.get("Content-Type", "")
    if response.status_code == 200 and "audio" in content_type:
        print("TTS Success: Audio received.")
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(response.content)
        return out_path
    else:
        print(f"TTS Failed: Status {response.status_code}, Content: {response.text[:100]}")
        # Return None so the UI doesn't try to play a broken file
        return None

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None
    
    # 1. Transcribe
    asr_resp = requests.post(WHISPER_URL, headers=headers, data=open(audio_path, "rb").read())
    transcript = asr_resp.json().get("text", "ASR Error")

    # 2. Translate
    payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_DATA[input_lang]["mbart"],
            "tgt_lang": LANG_DATA[target_lang]["mbart"]
        }
    }
    tr_resp = requests.post(TRANSLATE_URL, headers=headers, json=payload)
    translation = tr_resp.json()[0].get("translation_text", "Translation Error")
    
    # 3. TTS
    audio_out = text_to_speech(translation, target_lang)
    return transcript, translation, audio_out

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 VOXTRAL v2 (with Debugging)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Input")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    with gr.Row():
        audio_out = gr.Audio(label="Output Audio", autoplay=True)
        speak_btn = gr.Button("🔊 Speak Translation")

    audio_in.stop_recording(translate_speech, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])
    speak_btn.click(text_to_speech, [trn_out, out_lang], audio_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)