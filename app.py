import gradio as gr
from huggingface_hub import InferenceClient
import os
import sys

# Force output to logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Initialize Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def translate_speech(audio_path):
    if not audio_path:
        return "No audio recorded.", ""
    
    try:
        print(f"Processing audio from: {audio_path}")
        
        # 1. Transcription (Whisper)
        asr_result = client.automatic_speech_recognition(
            audio_path, 
            model="openai/whisper-large-v3-turbo"
        )
        transcript = asr_result.text
        print(f"Transcript: {transcript}")

        # 2. Translation (Helsinki-NLP is more stable on free API than NLLB)
        # It automatically handles English to French
        translation_result = client.translation(
            transcript,
            model="Helsinki-NLP/opus-mt-en-fr"
        )
        translated_text = translation_result.translation_text
        print(f"Translation: {translated_text}")
        
        return transcript, translated_text

    except Exception as e:
        # If it's a 'NoneType' or empty error, we catch it here
        error_msg = str(e) if str(e) else "Unknown API Error (Empty response)"
        print(f"DETAILED ERROR: {error_msg}")
        return f"Error: {error_msg}", ""

# Gradio 6.0 UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Speech-to-French Prototype")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record")
    
    with gr.Column():
        out_transcription = gr.Textbox(label="Transcription (English)")
        out_translation = gr.Textbox(label="Translation (French)")
    
    # Triggering on 'change' or 'stop_recording'
    audio_in.stop_recording(translate_speech, inputs=audio_in, outputs=[out_transcription, out_translation])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)