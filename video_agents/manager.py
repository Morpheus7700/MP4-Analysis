import os
import streamlit as st
from video_agents.audio_agent import AudioAgent
from video_agents.visual_agent import VisualAgent
from video_agents.report_agent import ReportAgent
from utils.video_utils import VideoUtils
import tempfile

class ManagerAgent:
    def __init__(self):
        self.audio_agent = AudioAgent()
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

        # 3. Audio Extraction & Analysis
        st.info("Manager Agent: Delegating to Audio Agent...")
        # Use os.path to safely create the wav path
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        VideoUtils.extract_audio(video_path, audio_path)
        
        audio_report = self.audio_agent.analyze(audio_path)
        st.success("Audio Agent: Analysis complete.")
        with st.expander("See Audio Logs"):
            st.json(audio_report)

        # 4. Visual Extraction & Analysis
        st.info("Manager Agent: Delegating to Visual Agent...")
        # Extract frames (1 frame every 2 seconds to save time/compute)
        frames = VideoUtils.extract_frames(video_path, "frames_temp", interval=2)
        
        visual_report = self.visual_agent.analyze(frames)
        st.success("Visual Agent: Analysis complete.")
        with st.expander("See Visual Logs"):
            st.json(visual_report)
            if "captions" in visual_report:
                st.write("**Scene Captions:**")
                st.write(visual_report["captions"])
            if "timeline" in visual_report:
                st.write("**Frame Analysis Timeline:**")
                st.dataframe(visual_report["timeline"])

        # 5. Synthesis
        st.info("Manager Agent: Requesting Final Report...")
        final_report = self.report_agent.synthesize(audio_report, visual_report, meta)
        
        # Add raw data for UI debugging
        final_report["visual_raw"] = visual_report
        final_report["audio_raw"] = audio_report
        
        # Cleanup
        try:
            os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
            
        return final_report
