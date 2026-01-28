import streamlit as st
from video_agents.manager import ManagerAgent
import os

st.set_page_config(page_title="Video AI Analyst", layout="wide")

st.title("🧠 Autonomous Video AI Analyst (Deep Learning Edition)")
st.markdown("""
This system uses a team of **Advanced Deep Learning Agents** to analyze your video files.
- **Manager Agent**: Orchestrates the multi-modal pipeline.
- **Audio Agent (Whisper)**: High-fidelity speech-to-text and sound analysis.
- **Visual Agent (YOLO/BLIP/ViT)**: Objects, Actions, Emotions, and Gestures.
- **Report Agent (Gemma-2)**: Advanced LLM reasoning to synthesize intelligence reports.
""")

uploaded_file = st.file_uploader("Upload a Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("Start Deep Analysis"):
        manager = ManagerAgent()
        
        with st.spinner("🤖 Agents are collaborating (Loading Deep Learning Models)..."):
            report = manager.process_video(uploaded_file)
        
        st.divider()
        st.header(report["title"])
        
        st.markdown(report["narrative"])
        
        with st.expander("🛠️ Debug: Raw Agent Logs"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Visual Agent Data")
                st.json(report.get("visual_raw", {}))
            with col2:
                st.subheader("Audio Agent Data")
                st.json(report.get("audio_raw", {}))

        st.subheader("🔍 Supplemental Insights")
        for finding in report["key_findings"]:
            st.info(finding)

st.sidebar.title("System Status")
st.sidebar.success("All DL Models Active")
st.sidebar.info("Using Whisper (Audio) & Gemma-2 (LLM)")
st.sidebar.markdown("---")
st.sidebar.write("Running in **Local GPU/CPU Mode**")
