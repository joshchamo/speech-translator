import gradio as gr
import os
import torch
from huggingface_hub import InferenceClient
from datasets import load_dataset

# Initialize the 2026-standard Inference Client
# This handles the router logic automatically
client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Models we know are currently pinned and high-availability
STT_MODEL = "openai/whisper-large-v3-turbo"
TRANSLATE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
TTS_MODEL = "microsoft/speecht5_tts"

# Speaker embeddings are required for SpeechT5 to define the 'voice'
# We'll pull a standard one from the official dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

LANG_DATA = {
    "English": "en_XX", "Spanish": "es_XX", "French": "fr_XX", 
    "German": "de_DE", "Japanese": "ja_XX", "Chinese": "zh_CN"
}

def translate_and_speak(audio_path, in_lang, out_lang):
    if not audio_path: return "", "", None
    
    # 1. Faster Transcription using InferenceClient
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    # Client.automatic_speech_recognition is the 2026 preferred method
    transcript = client.automatic_speech_recognition(audio_data, model=STT_MODEL).text

    # 2. Translation
    payload = {
        "inputs": transcript,
        "parameters": {"src_lang": LANG_DATA[in_lang], "tgt_lang": LANG_DATA[out_lang]}
    }
    # Direct post for MBART as it's a specific translation task
    res_tr = client.post(json=payload, model=TRANSLATE_MODEL)
    import json
    translation = json.loads(res_tr.decode())[0]['translation_text']

    # 3. Stable TTS (SpeechT5)
    # This model is a 'pinned' service and rarely returns 404
    audio_output = client.text_to_speech(translation, model=TTS_MODEL)
    
    out_path = "output.wav"
    with open(out_path, "wb") as f:
        f.write(audio_output)
        
    return transcript, translation, out_path

# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌎 Universal Voxtral v4")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Voice Input")
        with gr.Column():
            in_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="English", label="Input Language")
            out_lang = gr.Dropdown(choices=list(LANG_DATA.keys()), value="French", label="Target Language")
    
    with gr.Row():
        txt_out = gr.Textbox(label="Transcription")
        trn_out = gr.Textbox(label="Translation")
    
    audio_out = gr.Audio(label="Voice Output", autoplay=True)
    
    # The action trigger
    audio_in.stop_recording(
        translate_and_speak, 
        inputs=[audio_in, in_lang, out_lang], 
        outputs=[txt_out, trn_out, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)