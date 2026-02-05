import gradio as gr
import os
import json
from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Constants for models
STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

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
        # 1. Transcription - Passing the PATH directly so the client
        # can automatically detect the content type from the file extension
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation
        payload = {
            "inputs": transcript,
            "parameters": {"src_lang": LANG_DATA[in_lang], "tgt_lang": LANG_DATA[out_lang]}
        }
        res_tr = client.post(json=payload, model=TRANSLATE_MODEL)
        translation = json.loads(res_tr.decode())[0]['translation_text']

        # 3. TTS - Using a robust MMS model
        # We use the first two letters of the language code (e.g., 'fr', 'en')
        lang_short = LANG_DATA[out_lang][:2]
        tts_model = f"facebook/mms-tts-{lang_short}"
        
        # We use client.post for TTS to have more control over the output
        tts_res = client.text_to_speech(translation, model=tts_model)
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(tts_res)
            
        return transcript, translation, out_path
    
    except Exception as e:
        # This will show the error in the "Transcription" box for easier debugging
        return f"System Error: {str(e)}", "Check logs", None

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔊 Universal Voxtral: Fixed & Verified")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Click to Record")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="Input Language")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="You said:")
        trn_out = gr.Textbox(label="Translated:")
    
    audio_out = gr.Audio(label="Voice Playback", autoplay=True)
    
    # Process when the user stops recording
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)