import whisper
import os
import torch

class AudioAgent:
    def __init__(self, model_size="base"):
        """
        Deep Learning Audio Agent using OpenAI Whisper.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Audio Agent: Loading Whisper '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def analyze(self, audio_path):
        """
        Transcribes audio using a local Deep Learning model.
        Supports 99 languages including Bengali, Hindi, Spanish, French, German, etc.
        """
        if not os.path.exists(audio_path):
            return {"status": "error", "message": "Audio file not found."}

        try:
            # Whisper can hallucinate on silence. We use specific thresholds.
            # no_speech_threshold helps filter out noise.
            result = self.model.transcribe(
                audio_path, 
                task="transcribe",
                fp16=False if self.device == "cpu" else True,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )
            
            transcript = result["text"].strip()
            
            # Filter common Whisper hallucinations on silent/noisy audio
            hallucinations = ["thank you", "thanks for watching", "subtitle", "identify the speaker", "please subscribe"]
            if any(h in transcript.lower() for h in hallucinations) and len(transcript.split()) < 5:
                transcript = ""

            detected_lang = result.get("language", "unknown")
            
            return {
                "agent": "AudioAgent",
                "status": "success" if transcript else "no_speech",
                "transcript": transcript if transcript else "No clear speech detected.",
                "detected_language": detected_lang,
                "segments": result.get("segments", []),
                "insights": self._generate_insights(transcript, detected_lang)
            }
        except Exception as e:
            return {
                "agent": "AudioAgent",
                "status": "error",
                "message": str(e)
            }

    def _generate_insights(self, text, lang):
        if not text:
            return ["No verbal content detected."]
        
        insights = [f"Detected Language: {lang.upper()}"]
        insights.append(f"Transcription length: {len(text.split())} words.")
        return insights