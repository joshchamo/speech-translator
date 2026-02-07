import gradio as gr
import requests
import os
from gtts import gTTS

# 1. Configuration
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Models
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

# 2. Language Mappings
# mBART-50 Code (for Translation) -> gTTS Code (for Speech)
LANG_CODES = {
    "Arabic": {"mbart": "ar_AR", "gtts": "ar"},
    "Chinese": {"mbart": "zh_CN", "gtts": "zh-cn"},
    "Czech": {"mbart": "cs_CZ", "gtts": "cs"},
    "Dutch": {"mbart": "nl_XX", "gtts": "nl"},
    "English": {"mbart": "en_XX", "gtts": "en"},
    "Finnish": {"mbart": "fi_FI", "gtts": "fi"},
    "French": {"mbart": "fr_XX", "gtts": "fr"},
    "German": {"mbart": "de_DE", "gtts": "de"},
    "Hindi": {"mbart": "hi_IN", "gtts": "hi"},
    "Indonesian": {"mbart": "id_ID", "gtts": "id"},
    "Italian": {"mbart": "it_IT", "gtts": "it"},
    "Japanese": {"mbart": "ja_XX", "gtts": "ja"},
    "Korean": {"mbart": "ko_KR", "gtts": "ko"},
    "Polish": {"mbart": "pl_PL", "gtts": "pl"},
    "Portuguese": {"mbart": "pt_XX", "gtts": "pt"},
    "Russian": {"mbart": "ru_RU", "gtts": "ru"},
    "Spanish": {"mbart": "es_XX", "gtts": "es"},
    "Swedish": {"mbart": "sv_SE", "gtts": "sv"},
    "Turkish": {"mbart": "tr_TR", "gtts": "tr"},
    "Ukrainian": {"mbart": "uk_UA", "gtts": "uk"},
    "Vietnamese": {"mbart": "vi_VN", "gtts": "vi"}
}

def query_api(url, payload, is_audio_in=False):
    h = headers.copy()
    try:
        if is_audio_in:
            h["Content-Type"] = "audio/wav"
            with open(payload, "rb") as f: data = f.read()
            response = requests.post(url, headers=h, data=data, timeout=30)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=30)
            
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        print(f"Connection Exception: {e}")
        return None

def text_to_speech_gtts(text, language):
    if not text: return None
    try:
        # Look up the 2-letter code for gTTS
        lang_code = LANG_CODES[language]["gtts"]
        
        # Create the audio file using gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        filename = f"output_{lang_code}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"gTTS Error: {str(e)}")
        return None

def run_pipeline(audio_path, input_lang, target_lang):
    if not audio_path: return "", "", None, "No audio."

    # A. Transcribe (Whisper)
    asr_data = query_api(WHISPER_URL, audio_path, is_audio_in=True)
    if not asr_data or "text" not in asr_data:
        return "ASR Failed", "", None, str(asr_data)
    transcript = asr_data["text"]

    # B. Translate (mBART-50)
    tr_payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_CODES[input_lang]["mbart"],
            "tgt_lang": LANG_CODES[target_lang]["mbart"]
        }
    }
    tr_data = query_api(TRANSLATE_URL, tr_payload)
    
    try:
        translation = tr_data[0]['translation_text']
    except:
        return transcript, "Translation Failed", None, str(tr_data)

    # C. Speak (gTTS)
    audio_path = text_to_speech_gtts(translation, target_lang)
    
    return transcript, translation, audio_path, "Success"

# --- UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ VOXTRAL: Universal Translator")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources="microphone", type="filepath", label="Record Speech")
            in_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="English", label="Input Language")
            out_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="French", label="Target Language")
            btn_run = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            txt_out = gr.Textbox(label="Transcription")
            trn_out = gr.Textbox(label="Translation")
            audio_out = gr.Audio(label="Spoken Translation", interactive=False, autoplay=True)
            
    with gr.Accordion("Debug Logs", open=False):
        debug_out = gr.Textbox()

    # Event 1: Stop Recording -> Run
    audio_in.stop_recording(
        run_pipeline, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )
    
    # Event 2: Button Click -> Run
    btn_run.click(
        run_pipeline,
        inputs=[audio_in, in_lang, out_lang],
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)