import torch
from transformers import pipeline
import os
import streamlit as st

class LLMEngine:
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LLM Engine: Initializing on {self.device}...")

    def _load_model(self):
        if self.pipeline is None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                print(f"LLM Engine: Loading {self.model_id}...")
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Primary model failed ({e}). Loading reliable fallback (Phi-1.5)...")
                self.pipeline = pipeline("text-generation", model="microsoft/phi-1_5", device_map="auto", trust_remote_code=True)

    def generate_report(self, audio_data, visual_data, meta):
        st.write("📖 LLM Engine: Reading analysis data...")
        self._load_model()
        prompt = self._construct_prompt(audio_data, visual_data, meta)
        
        st.write("✍️ LLM Engine: Synthesizing Intelligence Briefing...")
        # Significantly richer parameters for detailed output
        outputs = self.pipeline(
            prompt,
            max_new_tokens=300, # Reduced for speed on CPU/low-VRAM
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        full_text = outputs[0]["generated_text"]
        # Extract only the content after "OFFICIAL INTELLIGENCE REPORT"
        marker = "OFFICIAL INTELLIGENCE REPORT"
        if marker in full_text:
            return full_text.split(marker)[-1].strip()
        return full_text[len(prompt):].strip()

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