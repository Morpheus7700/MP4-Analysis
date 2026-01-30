import torch
from transformers import pipeline
import os
import streamlit as st
import gc

class LLMEngine:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LLM Engine: Initializing on {self.device}...")

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
                print(f"Primary model failed ({e}). Loading fallback (Phi-1.5)...")
                self.pipeline = pipeline(
                    "text-generation", 
                    model="microsoft/phi-1_5", 
                    device_map=device_map,
                    trust_remote_code=True
                )

    def generate_report(self, audio_data, visual_data, meta):
        st.write("📖 LLM Engine: Reading analysis data...")
        
        # --- PRE-CHECK FOR SPARSE DATA ---
        # If there's barely any data, don't ask the LLM to hallucinate.
        visual_narrative = visual_data['stats'].get('visual_narrative', '')
        objs = visual_data['stats'].get('unique_objects', [])
        transcript = audio_data.get('transcript', '')
        
        # Check if we have meaningful data
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
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=150, 
                do_sample=False, # Use greedy search for higher factual accuracy
                repetition_penalty=1.2,
                pad_token_id=self.pipeline.tokenizer.pad_token_id,
                clean_up_tokenization_spaces=True,
                return_full_text=False 
            )
            
            # Extract text carefully
            if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                report_body = outputs[0]["generated_text"].strip()
            else:
                report_body = str(outputs[0]).strip()

            # Fallback for manual slicing if return_full_text didn't work as expected
            if prompt in report_body:
                report_body = report_body.replace(prompt, "").strip()

            # --- ANTI-HALLUCINATION FILTER ---
            # Truncate if the model starts generating textbook patterns or essays
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
