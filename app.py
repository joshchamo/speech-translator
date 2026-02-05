import gradio as gr
from huggingface_hub import InferenceClient
import os

# The 'token' parameter pulls the secret you set in the Space Settings
# If HF_TOKEN is missing, this will fail with the 401 error you saw.
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def process_audio(audio_path):
    if audio_path is None:
        return "Please record audio.", ""
        
    try:
        # 1. ASR - Using a model guaranteed to be on the serverless fleet
        transcription_result = client.automatic_speech_recognition(
            audio_path, 
            model="openai/whisper-large-v3-turbo"
        )
        transcript = transcription_result.text
        
        # 2. Translation
        translation_result = client.translation(
            transcript,
            model="facebook/nllb-200-distilled-600M",
            extra_params={"src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}
        )
        translation = translation_result.translation_text
        
        return transcript, translation
        
    except Exception as e:
        # Detailed error reporting helps debugging
        return f"Error: {str(e)}", ""

# UI Design
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Real-time Translation Prototype")
    
    with gr.Row():
        audio_input = gr.Audio(sources="microphone", type="filepath")
    
    with gr.Row():
        text_output = gr.Textbox(label="Transcription (English)")
        translation_output = gr.Textbox(label="Translation (French)")
    
    audio_input.stop_recording(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[text_output, translation_output]
    )

# FIX: In Gradio 6.0, the theme must be passed here, not in gr.Blocks()
demo.launch(theme=gr.themes.Soft())