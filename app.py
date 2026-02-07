import gradio as gr
import requests
import os
import sys

# 1. Logging Setup (Keeps logs visible in HF Spaces)
sys.stdout.reconfigure(line_buffering=True)

# 2. Configuration
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Models
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

# 3. Language Mappings
# mBART-50 requires "xx_XX" format
LANG_CODES = {
    "Arabic": "ar_AR", "Chinese": "zh_CN", "Czech": "cs_CZ", "Dutch": "nl_XX", "English": "en_XX",
    "Finnish": "fi_FI", "French": "fr_XX", "German": "de_DE", "Hindi": "hi_IN", "Indonesian": "id_ID",
    "Italian": "it_IT", "Japanese": "ja_XX", "Korean": "ko_KR", "Polish": "pl_PL", "Portuguese": "pt_XX",
    "Russian": "ru_RU", "Spanish": "es_XX", "Swedish": "sv_SE", "Turkish": "tr_TR", "Ukrainian": "uk_UA",
    "Vietnamese": "vi_VN"
    # (You can add the rest of the 50 here, but these are the most likely to have active TTS models)
}

# MMS-TTS requires ISO-639-3 (3-letter) codes. 
# We map the mBART Name -> MMS Code here.
MMS_CODES = {
    "Arabic": "ara", "Chinese": "cmn", "Czech": "ces", "Dutch": "nld", "English": "eng",
    "Finnish": "fin", "French": "fra", "German": "deu", "Hindi": "hin", "Indonesian": "ind",
    "Italian": "ita", "Japanese": "jpn", "Korean": "kor", "Polish": "pol", "Portuguese": "por",
    "Russian": "rus", "Spanish": "spa", "Swedish": "swe", "Turkish": "tur", "Ukrainian": "ukr",
    "Vietnamese": "vie"
}

# 4. API Functions
def query_api(url, payload, is_audio_in=False, is_audio_out=False):
    h = headers.copy()
    try:
        if is_audio_in:
            h["Content-Type"] = "audio/wav"
            with open(payload, "rb") as f: data = f.read()
            response = requests.post(url, headers=h, data=data, timeout=30)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=30)
            
        # Handle 404/503 errors gracefully
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text[:50]}")
            return None
            
        if is_audio_out:
            return response.content # Return raw bytes for audio
        return response.json()
        
    except Exception as e:
        print(f"Connection Exception: {e}")
        return None

def text_to_speech(text, language):
    if not text: return None
    
    # 1. Get the correct 3-letter code
    mms_code = MMS_CODES.get(language)
    if not mms_code:
        print(f"No TTS code found for {language}")
        return None

    # 2. Construct the URL for that specific language model
    tts_url = f"https://router.huggingface.co/hf-inference/models/facebook/mms-tts-{mms_code}"
    print(f"Calling TTS: {tts_url}")

    # 3. Call API
    audio_bytes = query_api(tts_url, {"inputs": text}, is_audio_out=True)
    
    # 4. Save to file if successful
    if audio_bytes:
        path = f"output_{mms_code}.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path
    
    return None

# Main Pipeline
def run_pipeline(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None, "No audio."

    # A. Transcribe
    asr_data = query_api(WHISPER_URL, audio_path, is_audio_in=True)
    if not asr_data or "text" not in asr_data:
        return "ASR Failed", "", None, str(asr_data)
    transcript = asr_data["text"]

    # B. Translate
    tr_payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_CODES.get(input_lang, "en_XX"),
            "tgt_lang": LANG_CODES.get(target_lang, "fr_XX")
        }
    }
    tr_data = query_api(TRANSLATE_URL, tr_payload)
    
    # Parse Translation
    try:
        translation = tr_data[0]['translation_text']
    except:
        return transcript, "Translation Failed", None, str(tr_data)

    # C. Speak (TTS)
    # We call this immediately so it feels fast
    audio_path = text_to_speech(translation, target_lang)
    
    debug_info = f"Success. {input_lang} -> {target_lang}"
    if not audio_path:
        debug_info += " (TTS Model offline or busy)"

    return transcript, translation, audio_path, debug_info

# 5. UI Setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ VOXTRAL: Speak, Translate, Listen")
    
    with gr.Row():
        # Input Column
        with gr.Column():
            audio_in = gr.Audio(sources="microphone", type="filepath", label="Record Speech")
            in_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="English", label="Input Language")
            out_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="French", label="Target Language")
            btn_run = gr.Button("Translate", variant="primary")
        
        # Output Column
        with gr.Column():
            txt_out = gr.Textbox(label="Transcription")
            trn_out = gr.Textbox(label="Translation")
            # The Audio Player
            audio_out = gr.Audio(label="Spoken Translation", interactive=False, autoplay=True)
            # Extra Button just to speak text
            btn_speak = gr.Button("🔊 Re-Speak Translation")
            
    with gr.Accordion("System Logs", open=False):
        debug_out = gr.Textbox(label="Debug Info")

    # Logic Triggers
    # 1. When recording stops, run full pipeline
    audio_in.stop_recording(
        run_pipeline, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )
    
    # 2. Manual Translate Button
    btn_run.click(
        run_pipeline,
        inputs=[audio_in, in_lang, out_lang],
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )

    # 3. Manual Re-Speak Button (Only runs TTS)
    btn_speak.click(
        text_to_speech,
        inputs=[trn_out, out_lang],
        outputs=[audio_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        ssr_mode=False
    )