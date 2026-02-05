import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the client (Uses HF's free serverless infrastructure)
client = InferenceClient()

def process_audio(audio):
    # 1. Transcribe using Voxtral (via Inference API)
    # Note: Voxtral-Realtime is brand new; if the serverless API isn't 
    # hitting it yet, you can use 'openai/whisper-tiny' as a fast fallback.
    transcript = client.automatic_speech_recognition(audio, model="mistralai/Voxtral-Mini-4B-Realtime-2602").text
    
    # 2. Translate using NLLB-200
    # We specify the target language (e.g., French 'fra_Latn')
    translation = client.translation(
        transcript, 
        model="facebook/nllb-200-distilled-600M",
        extra_params={"src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}
    ).translation_text
    
    return transcript, translation

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources="microphone", type="filepath", streaming=False),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Translation (French)")],
    live=True # This makes the interface feel "real-time"
)

demo.launch()