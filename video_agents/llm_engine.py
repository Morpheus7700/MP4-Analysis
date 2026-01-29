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
        self._load_model()
        prompt = self._construct_prompt(audio_data, visual_data, meta)
        
        st.write("✍️ LLM Engine: Synthesizing Intelligence Briefing...")
        try:
            if self.pipeline.tokenizer.pad_token_id is None:
                self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=200, 
                do_sample=False, # Use greedy search for higher factual accuracy
                repetition_penalty=1.2,
                pad_token_id=self.pipeline.tokenizer.pad_token_id,
                clean_up_tokenization_spaces=True
            )
            
            full_text = outputs[0]["generated_text"]
            report_body = full_text[len(prompt):].strip()

            # --- ANTI-HALLUCINATION FILTER ---
            # Truncate if the model starts generating textbook patterns
            hallucination_triggers = ["Question:", "Question 1", "Example:", "Solution:", "Calculate", "Exercise:"]
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

        return f"""### INSTRUCTIONS:
Summarize the VIDEO DATA below into a short factual paragraph. 
- DO NOT invent stories or characters.
- DO NOT generate questions or math problems.
- Stick to the facts provided.

### EXAMPLE:
DATA: Duration 10s, Objects: person, chair. Visual: A person sits down. Audio: \"Hello\".
REPORT: The video shows a 10-second clip where a person is seen sitting on a chair and says \"Hello\".

### CURRENT VIDEO DATA:
- Duration: {meta.get('duration', 0):.1f}s
- Objects: {objs}
- Visual Events: {visual_narrative}
- Audio: {speech}

### FACTUAL REPORT:
"""