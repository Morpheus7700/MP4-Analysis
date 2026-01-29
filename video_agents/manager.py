import os
import streamlit as st
from video_agents.audio_agent import AudioAgent
from video_agents.visual_agent import VisualAgent
from video_agents.report_agent import ReportAgent
from utils.video_utils import VideoUtils
import tempfile

class ManagerAgent:
    def __init__(self, config=None):
        """
        Initializes the ManagerAgent with optional configuration.
        """
        self.config = config if config is not None else {"audio_model": "base", "llm_model": "google/gemma-2-2b-it"}
        
        # Initialize sub-agents with specific models if provided in config
        audio_model = self.config.get("audio_model", "base")
        self.audio_agent = AudioAgent(model_size=audio_model)
        
        self.visual_agent = VisualAgent()
        self.report_agent = ReportAgent()

    def process_video(self, video_file):
        """
        Orchestrates the entire video analysis process.
        """
        st.info("Manager Agent: Received video. Starting processing pipeline...")
        
        # 1. Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close() # CRITICAL: Close the file so other processes can access it on Windows
        video_path = tfile.name
        
        # 2. Extract Metadata
        meta = VideoUtils.get_video_info(video_path)
        st.write(f"**Manager Agent:** Video Duration: {meta['duration']:.2f}s, FPS: {meta['fps']}")

        # 3. Parallel Audio & Visual Analysis
        import concurrent.futures
        
        st.info("Manager Agent: Launching Parallel Neural Analysis...")
        
        def run_audio():
            audio_path = os.path.splitext(video_path)[0] + ".wav"
            if meta.get("has_audio"):
                if VideoUtils.extract_audio(video_path, audio_path):
                    return self.audio_agent.analyze(audio_path)
            return self.audio_agent.analyze(None)

        def run_visual():
            # Extract frames (Scene Detection is now default in VideoUtils)
            frames = VideoUtils.extract_frames(video_path, "frames_temp", interval=2)
            return self.visual_agent.analyze(frames)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_audio = executor.submit(run_audio)
            future_visual = executor.submit(run_visual)
            
            audio_report = future_audio.result()
            visual_report = future_visual.result()

        st.success("Manager Agent: Multi-modal Analysis Complete.")
        
        with st.expander("Explore Consolidated Intelligence Feed"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Visual Intelligence**")
                st.json(visual_report)
            with col2:
                st.write("**Audio Intelligence**")
                st.json(audio_report)

        # 4. Synthesis
        st.info("Manager Agent: Requesting Final Report Synthesis...")
        final_report = self.report_agent.synthesize(audio_report, visual_report, meta)
        
        # Add raw data for UI debugging
        final_report["visual_raw"] = visual_report
        final_report["audio_raw"] = audio_report
        
        # Cleanup
        try:
            os.remove(video_path)
            audio_path_tmp = os.path.splitext(video_path)[0] + ".wav"
            if os.path.exists(audio_path_tmp):
                os.remove(audio_path_tmp)
        except:
            pass
            
        return final_report
