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
            try:
                print(f"LLM Engine: Loading {self.model_id} on {self.device}...")
                # We use a very standard loading procedure to avoid 'meta' device bugs
                # on systems with limited VRAM/Accelerate issues.
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    model_kwargs={"low_cpu_mem_usage": False} # Force full load to avoid meta tensors
                )
            except Exception as e:
                print(f"Primary model failed ({e}). Loading fallback (Phi-1.5)...")
                self.pipeline = pipeline(
                    "text-generation", 
                    model="microsoft/phi-1_5", 
                    device=0 if self.device == "cuda" else -1,
                    trust_remote_code=True,
                    model_kwargs={"low_cpu_mem_usage": False}
                )

    def generate_report(self, audio_data, visual_data, meta):
        st.write("📖 LLM Engine: Reading analysis data...")
        self._load_model()
        prompt = self._construct_prompt(audio_data, visual_data, meta)
        
        st.write("✍️ LLM Engine: Synthesizing Intelligence Briefing (This may take 30-90s on CPU)...")
        # Significantly richer parameters for detailed output
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=200, # Further reduced for speed
                do_sample=True, 
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            full_text = outputs[0]["generated_text"]
            # Clear memory after inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Extract only the content after the report header
            marker = "OFFICIAL INTELLIGENCE REPORT:"
            if marker in full_text:
                return full_text.split(marker)[-1].strip()
            return full_text[len(prompt):].strip()
        except Exception as e:
            return f"Synthesis Engine encountered a delay: {str(e)}. Please try again with Turbo Mode or smaller video."

    def _construct_prompt(self, audio, visual, meta):
        # Explicit grounding data
        visual_narrative = visual['stats'].get('visual_narrative', 'No visual description available.')
        objs = ', '.join(visual['stats'].get('unique_objects', []))
        emos = ', '.join([f"{k} ({v})" for k, v in visual['stats'].get('total_emotions', {}).items()])
        gestures = ', '.join([f"{k} ({v})" for k, v in visual['stats'].get('total_gestures', {}).items()])
        
        transcript = audio.get('transcript', '').strip()
        if not transcript or "No clear speech" in transcript or "No audio file" in transcript:
            speech = "[No verbal communication detected]"
        else:
            speech = f"\"{transcript}\""

        # Professional Analyst Prompt
        return f"""<SYSTEM_INSTRUCTION>
You are a Senior Intelligence Analyst. Synthesize the provided multi-modal data into a HIGHLY STRUCTURED, CHRONOLOGICAL Intelligence Briefing.
- Use a professional, objective tone.
- Elaborate on the RELATIONSHIP between what is seen and what is heard.
- Describe the FLOW of events using temporal markers.
- If data seems contradictory, note the discrepancy neutrally.
- DO NOT use bullet points in the "Briefing Narrative" section; use flowing paragraphs.
</SYSTEM_INSTRUCTION>

RAW DATA FEED:
[METADATA] Duration: {meta.get('duration', 0):.1f}s | FPS: {meta.get('fps', 0)}
[VISUAL CHRONOLOGY] {visual_narrative}
[OBSERVED ENTITIES] {objs}
[EMOTIONAL STATE] {emos}
[GESTURES & INTERACTIONS] {gestures}
[AUDIO TRANSCRIPTION] {speech}

OFFICIAL INTELLIGENCE REPORT:
BRIEFING NARRATIVE:
"""