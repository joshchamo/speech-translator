import gradio as gr
import os
import json
import sys
from huggingface_hub import InferenceClient

# Force logs to print immediately
sys.stdout.reconfigure(line_buffering=True)

print("--- APP STARTING ---", flush=True)

# Test Token Presence
token = os.getenv("HF_TOKEN")
if not token:
    print("CRITICAL: HF_TOKEN is missing!", flush=True)

client = InferenceClient(token=token)

# Models
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
    print(f"DEBUG: Processing audio from {audio_path}", flush=True)
    
    try:
        # 1. Transcription
        print(f"DEBUG: Calling STT...", flush=True)
        asr_res = client.automatic_speech_recognition(audio_path, model=STT_MODEL)
        transcript = asr_res.text
        print(f"DEBUG: Transcript: {transcript}", flush=True)

        # 2. Translation
        print(f"DEBUG: Calling Translation...", flush=True)
        payload = {
            "inputs": transcript,
            "parameters": {"src_lang": LANG_DATA[in_lang], "tgt_lang": LANG_DATA[out_lang]}
        }
        # Using the most robust request method for 2026
        response = client.request(json=payload, model=TRANSLATE_MODEL)
        translation = json.loads(response.decode())[0]['translation_text']
        print(f"DEBUG: Translation: {translation}", flush=True)

        # 3. TTS
        lang_short = LANG_DATA[out_lang][:2]
        tts_model = f"facebook/mms-tts-{lang_short}"
        print(f"DEBUG: Calling TTS ({tts_model})...", flush=True)
        
        audio_content = client.text_to_speech(translation, model=tts_model)
        
        out_path = "output.wav"
        with open(out_path, "wb") as f:
            f.write(audio_content)
            
        print("DEBUG: Processing Complete.", flush=True)
        return transcript, translation, out_path
    
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}", flush=True)
        return f"Error: {str(e)}", "Check Logs", None

# --- UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ VOXTRAL v7 (Debug Mode)")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Speak")
        in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="From")
        out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="To")
    
    txt_out = gr.Textbox(label="Transcript")
    trn_out = gr.Textbox(label="Translation")
    audio_out = gr.Audio(label="Voice Output", autoplay=True)
    
    audio_in.stop_recording(translate_and_speak, [audio_in, in_lang, out_lang], [txt_out, trn_out, audio_out])

if __name__ == "__main__":
    print("--- LAUNCHING GRADIO ---", flush=True)
    # Gradio 6.0: theme and other params in launch()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft()
    )