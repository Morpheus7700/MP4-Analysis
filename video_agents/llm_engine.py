import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

class LLMEngine:
    """
    A Deep Learning inference engine that uses local LLMs (like Gemma-2-2b)
    to perform reasoning and synthesis across video analysis outputs.
    """
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"LLM Engine: Initializing on {self.device}...")

    def _load_model(self):
        if self.pipeline is None:
            # We use a 2B model which fits in ~5-6GB RAM/VRAM
            # In a real environment, we'd use bitsandbytes for quantization, 
            # but for standard CPU/GPU compat, we'll use half-precision or auto.
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                )
            except Exception as e:
                print(f"Error loading LLM: {e}. Falling back to a lighter model (distilgpt2).")
                self.pipeline = pipeline("text-generation", model="distilgpt2", device_map="auto")

    def generate_report(self, audio_data, visual_data, meta):
        """
        Synthesizes raw JSON data into a professional intelligence report.
        """
        self._load_model()
        
        # Prepare the context
        prompt = self._construct_prompt(audio_data, visual_data, meta)
        
        # Manual template that works with Gemma, GPT-2, and others
        prompt_formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Fallback for models that don't recognize Gemma tokens (like GPT-2)
        if "gpt2" in self.pipeline.model.name_or_path.lower():
            prompt_formatted = f"User: {prompt}\n\nAssistant: Here is the intelligence report:\n"

        outputs = self.pipeline(
            prompt_formatted,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            truncation=True,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        full_text = outputs[0]["generated_text"]
        # Extract only the newly generated part
        if prompt_formatted in full_text:
            return full_text[full_text.find(prompt_formatted) + len(prompt_formatted):].strip()
        return full_text.strip()

    def _construct_prompt(self, audio, visual, meta):
        # Concise Visual Summary
        vis = f"SCENE: {visual.get('captions', ['N/A'])[0]}\nOBJECTS: {', '.join(visual['stats'].get('unique_objects', []))}\nEMOTIONS: {visual['stats'].get('total_emotions', {})}"
        
        # Audio Context
        transcript = audio.get('transcript', '').strip()
        lang = audio.get('detected_language', 'unknown')
        speech = f"TRANSCRIPT ({lang}): {transcript}" if transcript and "No clear speech" not in transcript else "SPEECH: None detected."

        return f"""Task: Summarize this video data into a professional report.

[DATA]
{vis}
{speech}

[FORMAT]
1. Title
2. Summary: (3 sentences max)
3. Speech Insights: (Translation/Summary of dialogue)
4. Context: (Likely intent)

Report:
"""
