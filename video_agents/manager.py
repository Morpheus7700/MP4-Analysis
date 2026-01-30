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
        
        # Store Gemini API Key in environment if provided
        if self.config.get("gemini_api_key"):
            os.environ["GEMINI_API_KEY"] = self.config.get("gemini_api_key")
        
        # Initialize sub-agents with specific models if provided in config
        audio_model = self.config.get("audio_model", "base")
        self.audio_agent = AudioAgent(model_size=audio_model)
        
        self.visual_agent = VisualAgent()
        self.report_agent = ReportAgent(model_id=self.config.get("llm_model"))

    def _check_ffmpeg(self):
        import shutil
        if shutil.which("ffmpeg") is not None:
            return True
        try:
            import imageio_ffmpeg as ffmpeg_lib
            return os.path.exists(ffmpeg_lib.get_ffmpeg_exe())
        except:
            return False

    def process_video(self, video_file):
        """
        Orchestrates the entire video analysis process.
        """
        # Safe Status Handling
        try:
            status_manager = st.status("🧠 Agents are collaborating...", expanded=True)
        except (AttributeError, TypeError):
            class DummyStatus:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def update(self, **kwargs): pass
                def write(self, text): print(f"[STATUS] {text}")
            status_manager = DummyStatus()

        with status_manager as status:
            st.info("Manager Agent: Initializing Neural Pipeline...")
            
            # 1. Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close() 
            video_path = tfile.name
            
            # 2. Extract Metadata
            st.write("🔍 Analyzing Video Metadata...")
            meta = VideoUtils.get_video_info(video_path)
            st.write(f"📊 **Metadata:** {meta['duration']:.2f}s, {meta['fps']} FPS, Audio: {'Yes' if meta['has_audio'] else 'No'}")

            # 3. Parallel Audio & Visual Analysis
            import concurrent.futures
            
            has_ffmpeg = self._check_ffmpeg()
            if not has_ffmpeg:
                st.warning("⚠️ FFmpeg not found. Audio analysis will be skipped. Please install ffmpeg for full intelligence.")

            st.info("🚀 Launching Parallel Neural Agents...")
            
            def run_audio():
                if not has_ffmpeg or not meta.get("has_audio"):
                    return self.audio_agent.analyze(None)
                    
                audio_path = os.path.splitext(video_path)[0] + ".wav"
                st.write("🎙️ Audio Agent: Extracting & Transcribing...")
                if VideoUtils.extract_audio(video_path, audio_path):
                    return self.audio_agent.analyze(audio_path)
                return self.audio_agent.analyze(None)

            def run_visual():
                st.write("👁️ Visual Agent: Scanning frames (Batch Processing)...")
                frames = VideoUtils.extract_frames(video_path, "frames_temp", interval=2)
                st.write(f"📸 Visual Agent: Processing {len(frames)} keyframes...")
                return self.visual_agent.analyze(frames)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_audio = executor.submit(run_audio)
                future_visual = executor.submit(run_visual)
                
                audio_report = future_audio.result()
                visual_report = future_visual.result()

            st.success("Manager Agent: Multi-modal Analysis Complete.")
            
            # 4. Synthesis
            st.info("Manager Agent: Requesting Final Report Synthesis...")
            final_report = self.report_agent.synthesize(audio_report, visual_report, meta, video_path=video_path)
            
            # Cleanup moved after synthesis to ensure metadata is available
            try:
                os.remove(video_path)
                audio_path_tmp = os.path.splitext(video_path)[0] + ".wav"
                if os.path.exists(audio_path_tmp):
                    os.remove(audio_path_tmp)
            except:
                pass

            status.update(label="Neural Synthesis Complete!", state="complete")
            
        # Add raw data for UI debugging
        final_report["visual_raw"] = visual_report
        final_report["audio_raw"] = audio_report
            
        return final_report
