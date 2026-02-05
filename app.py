import gradio as gr
import requests
import os
import sys

# Immediate log flushing for HF Container tab
sys.stdout.reconfigure(line_buffering=True)

API_TOKEN = os.getenv("HF_TOKEN")
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
TRANSLATE_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Complete mapping for mBART-50's 50 supported languages
LANG_CODES = {
    "Arabic": "ar_AR", "Czech": "cs_CZ", "German": "de_DE", "English": "en_XX", "Spanish": "es_XX", 
    "Estonian": "et_EE", "Finnish": "fi_FI", "French": "fr_XX", "Gujarati": "gu_IN", "Hindi": "hi_IN", 
    "Italian": "it_IT", "Japanese": "ja_XX", "Kazakh": "kk_KZ", "Korean": "ko_KR", "Lithuanian": "lt_LT", 
    "Latvian": "lv_LV", "Burmese": "my_MM", "Nepali": "ne_NP", "Dutch": "nl_XX", "Romanian": "ro_RO", 
    "Russian": "ru_RU", "Sinhala": "si_LK", "Turkish": "tr_TR", "Vietnamese": "vi_VN", "Chinese": "zh_CN", 
    "Afrikaans": "af_ZA", "Azerbaijani": "az_AZ", "Bengali": "bn_IN", "Persian": "fa_IR", "Hebrew": "he_IL", 
    "Croatian": "hr_HR", "Indonesian": "id_ID", "Georgian": "ka_GE", "Khmer": "km_KH", "Macedonian": "mk_MK", 
    "Malayalam": "ml_IN", "Mongolian": "mn_MN", "Marathi": "mr_IN", "Polish": "pl_PL", "Pashto": "ps_AF", 
    "Portuguese": "pt_XX", "Swedish": "sv_SE", "Swahili": "sw_KE", "Tamil": "ta_IN", "Telugu": "te_IN", 
    "Thai": "th_TH", "Tagalog": "tl_XX", "Ukrainian": "uk_UA", "Urdu": "ur_PK", "Xhosa": "xh_ZA", 
    "Galician": "gl_ES", "Slovene": "sl_SI"
}

def query_api(url, payload, is_audio=False):
    h = headers.copy()
    try:
        if is_audio:
            h["Content-Type"] = "audio/wav"
            with open(payload, "rb") as f:
                data = f.read()
            response = requests.post(url, headers=h, data=data, timeout=20)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=20)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}", "details": response.text[:100]}
    except Exception as e:
        return {"error": "Connection Failed", "details": str(e)}

def translate_speech(audio_path, input_lang, target_lang):
    if not audio_path: return "No audio recorded.", "", "Please record something first."
    
    # 1. Transcription (Whisper-Turbo)
    # Variable fixed: asr_payload
    asr_payload = audio_path
    res_asr = query_api(WHISPER_URL, asr_payload, is_audio=True)
    
    if "error" in res_asr:
        return f"ASR Error: {res_asr['error']}", "", res_asr.get('details', '')
    
    transcript = res_asr.get("text", "")

    # 2. Translation (mBART-50)
    src_code = LANG_CODES[input_lang]
    tgt_code = LANG_CODES[target_lang]
    
    tr_payload = {
        "inputs": transcript,
        "parameters": {"src_lang": src_code, "tgt_lang": tgt_code}
    }
    
    res_tr = query_api(TRANSLATE_URL, tr_payload)
    
    try:
        # mBART-50 returns a list: [{'translation_text': '...'}]
        if isinstance(res_tr, list) and len(res_tr) > 0:
            translation = res_tr[0].get("translation_text", "Result field missing")
        else:
            translation = f"API Message: {res_tr.get('error', 'Model Loading or Offline')}"
    except:
        translation = "Error parsing translation."
        
    return transcript, translation, f"Used: {src_code} -> {tgt_code}"

# --- UI with All 50 Languages ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌎 VOXTRAL: Universal 50-Language Translator")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        with gr.Column():
            in_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="English", label="Speaking Language")
            out_lang = gr.Dropdown(choices=sorted(list(LANG_CODES.keys())), value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")

    with gr.Accordion("Debug Trace", open=False):
        debug_out = gr.Textbox(label="Details")

    audio_in.stop_recording(
        translate_speech, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, debug_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(), 
        ssr_mode=False
    )