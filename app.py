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
# Keys: Display Name
# Values: {'mbart': mBART_Code, 'gtts': gTTS_Code}
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
    "Hebrew": {"mbart": "he_IL", "gtts": "iw"}, # Note: gTTS uses 'iw' for Hebrew
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
            with open(payload, "rb") as f: data = f.read()
            # Whisper can be heavy, so we give it a longer timeout
            response = requests.post(url, headers=h, data=data, timeout=60)
        else:
            response = requests.post(url, headers=h, json=payload, timeout=30)