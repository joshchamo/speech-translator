import gradio as gr
import os
from huggingface_hub import InferenceClient

# Initialize the client (Standard for 2026)
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# High-availability models
STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
# Using a model that doesn't require local speaker embeddings for simplicity
TTS_MODEL = "facebook/mms-tts-fra" 

LANG_DATA = {
    "English": "en_XX", 
    "Spanish": "es_XX", 
    "French": "fr_XX", 
    "German": "de_DE", 
    "Japanese": "ja_XX"
}

def translate_and_speak(audio_path, in_lang, out_lang):
    if not audio_path: return "", "", None
    
    try:
        # 1. Transcription
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        asr_res = client.automatic_speech_recognition(audio_data, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation
        payload = {
            "inputs": transcript,
            "parameters": {"src_lang": LANG_DATA[in_lang], "tgt_lang": LANG_DATA[out_lang]}
        }
        # MBART is a conversational/translation model
        res_tr = client.post(json=payload, model=TRANSLATE_MODEL)
        import json
        translation = json.loads(res_tr.decode())[0]['translation_text']

        # 3. TTS (We call a specific language model to avoid 404s)
        # Note: In a production app, we'd dynamically change this model string
        tts_res = client.text_to_speech(translation, model=f"facebook/mms-tts-{LANG_DATA[out_lang][:2]}")
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(tts_res)
            
        return transcript, translation, out_path
    
    except Exception as e:
        return f"Error: {str(e)}", "", None

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌎 Universal Voxtral (Lightweight)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="In")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="Out")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Voice Output", autoplay=True)
    
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)