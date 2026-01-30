from video_agents.llm_engine import LLMEngine

class ReportAgent:
    def __init__(self, model_id=None):
        # The Report Agent now delegates the 'thinking' to the LLM Engine
        self.llm = LLMEngine(model_id=model_id) if model_id else LLMEngine()

    def synthesize(self, audio_report, visual_report, video_meta, video_path=None):
        """
        Uses a Deep Learning LLM to synthesize multi-modal data.
        """
        raw_report_text = self.llm.generate_report(audio_report, visual_report, video_meta, video_path=video_path)
        
        # Parse the LLM output into the structure expected by the UI
        # We can perform more complex parsing here if the LLM is instructed to return JSON
        
        final_summary = {
            "title": "Deep Learning Intelligence Briefing",
            "video_duration": f"{video_meta.get('duration', 0):.2f} seconds",
            "narrative": raw_report_text,
            "key_findings": visual_report.get("insights", []) + audio_report.get("insights", [])
        }

        return final_summary