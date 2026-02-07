import gradio as gr
import requests
import os
import sys
from gtts import gTTS

# 1. Logging & Config
sys.stdout.reconfigure(line_buffering=True)
API_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Models
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"

# 2. FULL 50-Language Mapping
LANG_CODES = {
    "Afrikaans": {"mbart": "af_ZA", "gtts": "af"},
    "Arabic": {"mbart": "ar_AR", "gtts": "ar"},
    "Azerbaijani": {"mbart": "az_AZ", "gtts": "az"},
    "Bengali": {"mbart": "bn_IN", "gtts": "bn"},
    "Burmese": {"mbart": "my_MM", "gtts": "my"},
    "Chinese": {"mbart": "zh_CN", "gtts": "zh-cn"},
    "Croatian": {"mbart": "hr_HR", "gtts": "hr"},
    "Czech": {"mbart": "cs_CZ", "gtts": "cs"},
    "Dutch": {"mbart": "nl_XX", "gtts": "nl"},
    "English": {"mbart": "en_XX", "gtts": "en"},
    "Estonian": {"mbart": "et_EE", "gtts": "et"},
    "Finnish": {"mbart": "fi_FI", "gtts": "fi"},
    "French": {"mbart": "fr_XX", "gtts": "fr"},
    "Galician": {"mbart": "gl_ES", "gtts": "gl"},
    "Georgian": {"mbart": "ka_GE", "gtts": "ka"},
    "German": {"mbart": "de_DE", "gtts": "de"},
    "Gujarati": {"mbart": "gu_IN", "gtts": "gu"},
    "Hebrew": {"mbart": "he_IL", "gtts": "iw"},
    "Hindi": {"mbart": "hi_IN", "gtts": "hi"},
    "Indonesian": {"mbart": "id_ID", "gtts": "id"},
    "Italian": {"mbart": "it_IT", "gtts": "it"},
    "Japanese": {"mbart": "ja_XX", "gtts": "ja"},
    "Kazakh": {"mbart": "kk_KZ", "gtts": "kk"},
    "Khmer": {"mbart": "km_KH", "gtts": "km"},
    "Korean": {"mbart": "ko_KR", "gtts": "ko"},
    "Latvian": {"mbart": "lv_LV", "gtts": "lv"},
    "Lithuanian": {"mbart": "lt_LT", "gtts": "lt"},
    "Macedonian": {"mbart": "mk_MK", "gtts": "mk"},
    "Malayalam": {"mbart": "ml_IN", "gtts": "ml"},
    "Marathi": {"mbart": "mr_IN", "gtts": "mr"},
    "Mongolian": {"mbart": "mn_MN", "gtts": "mn"},
    "Nepali": {"mbart": "ne_NP", "gtts": "ne"},
    "Pashto": {"mbart": "ps_AF", "gtts": "ps"},
    "Persian": {"mbart": "fa_IR", "gtts": "fa"},
    "Polish": {"mbart": "pl_PL", "gtts": "pl"},
    "Portuguese": {"mbart": "pt_XX", "gtts": "pt"},
    "Romanian": {"mbart": "ro_RO", "gtts": "ro"},
    "Russian": {"mbart": "ru_RU", "gtts": "ru"},
    "Sinhala": {"mbart": "si_LK", "gtts": "si"},
    "Slovene": {"mbart": "sl_SI", "gtts": "sl"},
    "Spanish": {"mbart": "es_XX", "gtts": "es"},
    "Swahili": {"mbart": "sw_KE", "gtts": "sw"},
    "Swedish": {"mbart": "sv_SE", "gtts": "sv"},
    "Tagalog": {"mbart": "tl_XX", "gtts": "tl"},
    "Tamil": {"mbart": "ta_IN", "gtts": "ta"},
    "Telugu": {"mbart": "te_IN", "gtts": "te"},
    "Thai": {"mbart": "th_TH", "gtts": "th"},
    "Turkish": {"mbart": "tr_TR", "gtts": "tr"},
    "Ukrainian": {"mbart": "uk_UA", "gtts": "uk"},
    "Urdu": {"mbart": "ur_PK", "gtts": "ur"},
    "Vietnamese": {"mbart": "vi_VN", "gtts": "vi"},
    "Xhosa": {"mbart": "xh_ZA", "gtts": "xh"}
}

def query_api(url, payload, is_audio_in=False):
    h = headers.copy()
    try:
        if is_audio_in:
            h["Content-Type"] = "audio/wav"
            with open(payload, "rb") as f:
                data = f.read()
            response = requests.post(url, headers=h, data=data, timeout=60)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=30)
            
        if response.status_code != 200:
            return {"error": f"API {response.status_code}", "text": response.text}
        return response.json()
    except Exception as e:
        return {"error": "Connection Error", "text": str(e)}

def text_to_speech_gtts(text, language):
    if not text: return None, "No text to speak."
    try:
        lang_code = LANG_CODES.get(language, {}).get("gtts", "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)
        filename = f"output_{lang_code}.mp3"
        tts.save(filename)
        return filename, "Audio generated successfully."
    except Exception as e:
        return None, f"TTS Error: {str(e)}"

def run_pipeline(audio_path, input_lang, target_lang):
    if not audio_path: 
        return "", "", None, "Please record audio first."

    # 1. Transcribe
    asr_data = query_api(WHISPER_URL, audio_path, is_audio_in=True)
    if "error" in asr_data:
        return "ASR Failed", "", None, f"Whisper Error: {asr_data.get('text', 'Unknown error')}"
    
    transcript = asr_data.get("text", "")
    if not transcript:
        return "No speech detected", "", None, "Whisper returned empty text."

    # 2. Translate
    tr_payload = {
        "inputs": transcript,
        "parameters": {
            "src_lang": LANG_CODES[input_lang]["mbart"],
            "tgt_lang": LANG_CODES[target_lang]["mbart"]
        }
    }
    tr_data = query_api(TRANSLATE_URL, tr_payload)
    
    if isinstance(tr_data, dict) and "error" in tr_data:
         return transcript, "Translation Failed", None, f"mBART Error: {tr_data.get('text', 'Unknown error')}"
         
    try:
        translation = tr_data[0]['translation_text']
    except:
        return transcript, "Translation Parsing Failed", None, str(tr_data)

    # 3. Speak
    audio_path, tts_msg = text_to_speech_gtts(translation, target_lang)
    
    return transcript, translation, audio_path, f"Done. {tts_msg}"

# --- UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 VOXTRAL: 50-Language Voice Translator")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources="microphone", type="filepath", label="1. Record")
            with gr.Row():
                in_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="English", label="Input Language")
                out_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="Spanish", label="Target Language")
            btn_run = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            txt_out = gr.Textbox(label="Transcription")
            trn_out = gr.Textbox(label="Translation")
            audio_out = gr.Audio(label="Spoken Result", interactive=False, autoplay=True)
            
    with gr.Accordion("Debug Logs", open=False):
        debug_out = gr.Textbox(label="System Status")

    # Pipeline Triggers
    audio_in.stop_recording(
        run_pipeline, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )
    
    btn_run.click(
        run_pipeline,
        inputs=[audio_in, in_lang, out_lang],
        outputs=[txt_out, trn_out, audio_out, debug_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)