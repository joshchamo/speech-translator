import gradio as gr
from huggingface_hub import InferenceClient
import os
import sys

# Force logs to appear immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Initialize the client with your secret token
# If HF_TOKEN is missing in Settings, this will use guest limits
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def translate_speech(audio):
    if audio is None:
        return "No audio recorded.", ""
    
    try:
        # 1. ASR (Speech to Text)
        # Using a reliable model for the free Inference API
        asr_output = client.automatic_speech_recognition(
            audio, 
            model="openai/whisper-large-v3-turbo"
        )
        text = asr_output.text
        
        # 2. Translation (Text to French)
        # We pass the src/tgt codes directly as part of the model string or data
        # Most NLLB instances on HF API default to English if not specified, 
        # but let's use the standard translation task.
        translation = client.translation(
            text,
            model="facebook/nllb-200-distilled-600M"
        ).translation_text
        
        return text, translation

    except Exception as e:
        print(f"Error encountered: {e}")
        return f"Error: {str(e)}", ""

# Define the UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Real-time Prototype")
    
    with gr.Row():
        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record Speech")
    
    with gr.Column():
        transcription = gr.Textbox(label="Transcription")
        translation = gr.Textbox(label="Translation")
    
    # Process when the user stops recording
    audio_in.stop_recording(translate_speech, inputs=audio_in, outputs=[transcription, translation])

# IMPORTANT: server_name and port are required for HF Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)