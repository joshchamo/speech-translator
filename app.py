import gradio as gr
import os
import json
from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# High-availability models
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
        # 1. Transcription
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text

        # 2. Translation (Using the correct task-based method)
        # We use .request because MBART is a specific seq2seq model
        src_code = LANG_DATA[in_lang]
        tgt_code = LANG_DATA[out_lang]
        
        # New 2026 syntax for model requests
        payload = {
            "inputs": transcript,
            "parameters": {"src_lang": src_code, "tgt_lang": tgt_code}
        }
        
        # Using headers to solve the previous 'Content-Type' issue
        response = client.request(json=payload, model=TRANSLATE_MODEL)
        translation = json.loads(response.decode())[0]['translation_text']

        # 3. TTS
        lang_short = tgt_code[:2]
        tts_model = f"facebook/mms-tts-{lang_short}"
        
        audio_content = client.text_to_speech(translation, model=tts_model)
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(audio_content)
            
        return transcript, translation, out_path
    
    except Exception as e:
        return f"Error: {str(e)}", "Please try again", None

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ VOXTRAL v5 (Stable API)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_CODES.keys()), value="English", label="Source Language")
            out_lang = gr.Dropdown(choices=list(LANG_CODES.keys()), value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcript")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Audio Result", autoplay=True)
    
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)