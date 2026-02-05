def process_audio(audio_path):
    if audio_path is None:
        return "Please record audio.", ""
        
    try:
        # 1. ASR - Transcription
        transcription_result = client.automatic_speech_recognition(
            audio_path, 
            model="openai/whisper-large-v3-turbo"
        )
        transcript = transcription_result.text
        
        # 2. Translation - Using .post() for precise parameter control
        # This sends a direct JSON payload to the model
        payload = {
            "inputs": transcript,
            "parameters": {
                "src_lang": "eng_Latn", 
                "tgt_lang": "fra_Latn"
            }
        }
        
        # Use client.post to send the request to NLLB-200
        response = client.post(
            json=payload,
            model="facebook/nllb-200-distilled-600M",
            task="translation"
        )
        
        # Parse the JSON response
        import json
        result = json.loads(response.decode("utf-8"))
        translation = result[0]['translation_text']
        
        return transcript, translation
        
    except Exception as e:
        return f"Error: {str(e)}", ""