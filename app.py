import gradio as gr
from huggingface_hub import InferenceClient
import os

# 1. Initialize the client. 
# Make sure you have added HF_TOKEN to your Space's 'Settings > Secrets'
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def process_audio(audio_path):
    # Guard clause: stop if no audio is received
    if audio_path is None:
        return "Please record some audio first.", ""
        
    try:
        # A. Transcription: Using Whisper Large v3 Turbo (highly reliable on Serverless API)
        transcription_result = client.automatic_speech_recognition(
            audio_path, 
            model="openai/whisper-large-v3-turbo"
        )
        transcript = transcription_result.text
        
        # B. Translation: Using NLLB-200
        # NLLB requires 'eng_Latn' (English) and 'fra_Latn' (French) codes
        translation_result = client.translation(
            transcript,
            model="facebook/nllb-200-distilled-600M",
            extra_params={"src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}
        )
        translation = translation_result.translation_text
        
        return transcript, translation
        
    except Exception as e:
        # This will show the actual error message in the UI if it fails again
        return f"ASR Error: {str(e)}", f"Translation Error: {str(e)}"

# 2. Build a clean UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Fast Speech-to-French Translator")
    gr.Markdown("Click the microphone, speak, and then click 'Stop' or wait for processing.")
    
    with gr.Row():
        # type='filepath' is essential for sending data to the API correctly
        audio_input = gr.Audio(sources=["microphone"], type="filepath")
    
    with gr.Column():
        text_output = gr.Textbox(label="1. Transcription (English)", placeholder="Waiting for speech...")
        translation_output = gr.Textbox(label="2. Translation (French)", placeholder="Translation will appear here...")

    # We use 'stop_recording' as the trigger to keep things simple
    audio_input.stop_recording(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[text_output, translation_output]
    )

demo.launch()