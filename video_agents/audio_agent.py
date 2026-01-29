import whisper
import os
import torch

class AudioAgent:
    def __init__(self, model_size="base"):
        """
        Deep Learning Audio Agent using OpenAI Whisper.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        print(f"Audio Agent: Loading Whisper '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def analyze(self, audio_path):
        """
        Transcribes audio using a local Deep Learning model.
        Supports 99 languages including Bengali, Hindi, Spanish, French, German, etc.
        """
        if not audio_path or not os.path.exists(audio_path):
            return {"status": "no_audio", "transcript": "No audio file available for analysis.", "insights": ["Audio analysis skipped: No audio track found."]}

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
            hallucinations = ["thank you", "thanks for watching", "subtitle", "identify the speaker", "please subscribe", "watching"]
            if any(h in transcript.lower() for h in hallucinations) and len(transcript.split()) < 4:
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
            error_msg = str(e)
            if "WinError 2" in error_msg or "ffmpeg" in error_msg.lower():
                print("Audio Agent: FFmpeg missing or inaccessible.")
                return {
                    "agent": "AudioAgent",
                    "status": "error",
                    "transcript": "Transcription skipped: FFmpeg utility is missing on this system.",
                    "insights": ["Please install FFmpeg to enable audio analysis."]
                }
            print(f"Audio Agent Error: {e}")
            return {
                "agent": "AudioAgent",
                "status": "error",
                "transcript": "Audio analysis failed due to a processing error.",
                "message": str(e),
                "insights": ["Audio module encountered an internal error."]
            }

    def _generate_insights(self, text, lang):
        if not text:
            return ["No verbal content detected."]
        
        insights = [f"Detected Language: {lang.upper()}"]
        insights.append(f"Transcription length: {len(text.split())} words.")
        return insights