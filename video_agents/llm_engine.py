import torch
from transformers import pipeline
import os
import streamlit as st
import gc
import google.generativeai as genai

class LLMEngine:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # We'll check for the API key dynamically during report generation
        print(f"LLM Engine: Initializing with target model '{self.model_id}' on {self.device}...")

    def _load_model(self):
        if self.pipeline is None:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            # Use device_map="auto" for more robust weight loading
            device_map = "auto" if self.device == "cuda" else None
            
            try:
                print(f"LLM Engine: Loading {self.model_id} on {self.device}...")
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Primary model failed ({e}). Loading fallback (Qwen 2.5 1.5B)...")
                self.pipeline = pipeline(
                    "text-generation", 
                    model="Qwen/Qwen2.5-1.5B-Instruct", 
                    device_map=device_map,
                    trust_remote_code=True
                )

    def generate_report(self, audio_data, visual_data, meta, video_path=None):
        """
        Generates a report using Gemini if an API key is provided, 
        otherwise falls back to the local LLM.
        """
        # 1. Check for Gemini API Key (Priority 1: Manager Config/App Input, Priority 2: ENV)
        # In this architecture, we check if manager passed it through st.session_state or direct config
        # But since ReportAgent/LLMEngine are recreated/called, we check st.session_state or os.environ
        api_key = os.environ.get("GEMINI_API_KEY")
        
        # If we have a video and an API key, use the superior Gemini 2.0 Flash
        if api_key and video_path and os.path.exists(video_path):
            return self._generate_with_gemini(video_path, audio_data, visual_data, api_key)

        # 2. Fallback to Local LLM
        return self._generate_local_report(audio_data, visual_data, meta)

    def _generate_with_gemini(self, video_path, audio_data, visual_data, api_key):
        st.write("💎 LLM Engine: Uploading to Gemini 2.0 Flash for Deep Understanding...")
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Use the data we already have to guide Gemini
            local_context = f"""
            Local Analysis Hints:
            - Objects detected locally: {', '.join(visual_data['stats'].get('unique_objects', []))}
            - Local Transcription: {audio_data.get('transcript', 'None')}
            """
            
            prompt = f"""Analyze this video and provide a professional Intelligence Briefing.
            {local_context}
            
            Instructions:
            - Describe key scenes, people, and actions.
            - Provide a narrative summary of the content.
            - Be precise and factual. Avoid generic descriptions.
            - If there are people, describe their interactions and setting.
            """
            
            video_file = genai.upload_file(video_path)
            
            # Poll for file to be ready (simplistic for this version)
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            response = model.generate_content([prompt, video_file])
            
            # Clean up
            genai.delete_file(video_file.name)
            
            return response.text
        except Exception as e:
            st.error(f"Gemini Analysis Failed: {e}. Falling back to local model...")
            return self._generate_local_report(audio_data, visual_data, {"duration": 0}) # Dummy meta for fallback

    def _generate_local_report(self, audio_data, visual_data, meta):
        st.write("📖 LLM Engine: Reading analysis data...")
        
        # --- PRE-CHECK FOR SPARSE DATA ---
        visual_narrative = visual_data['stats'].get('visual_narrative', '')
        objs = visual_data['stats'].get('unique_objects', [])
        transcript = audio_data.get('transcript', '')
        
        has_visuals = len(objs) > 0 or (visual_narrative and "unavailable" not in visual_narrative)
        has_audio = transcript and "No" not in transcript and len(transcript) > 5
        
        if not has_visuals and not has_audio:
            return "The analysis could not extract sufficient visual or audio data to generate a detailed narrative. Please ensure the video has clear content."

        self._load_model()
        prompt = self._construct_prompt(audio_data, visual_data, meta)
        
        st.write("✍️ LLM Engine: Synthesizing Intelligence Briefing...")
        try:
            if self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=150, 
                do_sample=False, 
                repetition_penalty=1.2,
                pad_token_id=self.pipeline.tokenizer.pad_token_id,
                clean_up_tokenization_spaces=True,
                return_full_text=False 
            )
            
            if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                report_body = outputs[0]["generated_text"].strip()
            else:
                report_body = str(outputs[0]).strip()

            if prompt in report_body:
                report_body = report_body.replace(prompt, "").strip()

            hallucination_triggers = [
                "Question:", "Question 1", "Example:", "Solution:", "Calculate", "Exercise:", 
                "Title:", "Introduction:", "Chapter", "Section 1", "Step 1:", "References:", "##"
            ]
            for trigger in hallucination_triggers:
                if trigger in report_body:
                    report_body = report_body.split(trigger)[0].strip()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return report_body if len(report_body) > 10 else "The video analysis was completed, but the narrative synthesis was inconclusive based on the raw data feed."

        except Exception as e:
            return f"Synthesis Engine encountered a delay: {str(e)}."

    def _construct_prompt(self, audio, visual, meta):
        visual_narrative = visual['stats'].get('visual_narrative', 'No visual description available.')
        objs = ', '.join(visual['stats'].get('unique_objects', []))
        
        transcript = audio.get('transcript', '').strip()
        speech = f"\"{transcript}\"" if transcript and "No" not in transcript else "[No speech]"

        return f"""Task: Summarize the following video analysis data into one concise, factual paragraph. Do not add outside information.

Data:
- Duration: {meta.get('duration', 0):.1f}s
- Objects Detected: {objs if objs else "None"}
- Visual Events: {visual_narrative}
- Audio/Speech: {speech}

Summary: The video is {meta.get('duration', 0):.1f} seconds long."""