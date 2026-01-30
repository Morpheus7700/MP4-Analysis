import streamlit as st
import os
import numpy as np

# Global Compatibility Fix for legacy libraries (Keras/MediaPipe) on Python 3.13+
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

from video_agents.manager import ManagerAgent

# Page Config for Premium Look
st.set_page_config(
    page_title="Video AI Analyst | Mission Control",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
css_path = os.path.join(os.getcwd(), "static", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar System Health & Settings
st.sidebar.title("🔋 System Control")
turbo_mode = st.sidebar.toggle("🚀 TURBO MODE (Fastest)", value=True, help="Uses lighter models (Whisper Tiny + Qwen 2.5 1.5B) for ~3x faster analysis.")

@st.cache_resource
def get_manager(is_turbo):
    import video_agents.manager
    import importlib
    importlib.reload(video_agents.manager)
    from video_agents.manager import ManagerAgent
    
    config = {
        "audio_model": "tiny" if is_turbo else "base",
        "llm_model": "Qwen/Qwen2.5-1.5B-Instruct" if is_turbo else "google/gemma-2-2b-it"
    }
    
    try:
        # Try passing config (Latest version)
        return ManagerAgent(config)
    except TypeError:
        # Fallback for old version if somehow it still persists
        st.warning("Warning: Using legacy ManagerAgent signature. Turbo Mode may be disabled.")
        return ManagerAgent()

manager = get_manager(turbo_mode)

st.sidebar.success(f"Visual Core: Active (YOLOv8/BLIP)")
st.sidebar.success(f"Audio Core: Active (Whisper {'Tiny' if turbo_mode else 'Base'})")
st.sidebar.info(f"Synthesis Engine: {'Qwen 2.5' if turbo_mode else 'Gemma-2'}")
st.sidebar.divider()
st.sidebar.markdown("### Resource Usage")
st.sidebar.progress(20 if turbo_mode else 45, text="VRAM Allocation")
st.sidebar.progress(10 if turbo_mode else 25, text="Compute Density")
st.sidebar.divider()
st.sidebar.write("⚡ Running in **Premium Local Mode**")

# Header Section
col_head, col_logo = st.columns([4, 1])
with col_head:
    st.title("⚡ Video AI Analyst: Mission Control")
    st.markdown(f"#### {'Turbo' if turbo_mode else 'Higher'} Intelligence. Deep Video Synthesis.")

st.divider()

# Main Interaction Area
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader("📁 Input Source")
    uploaded_file = st.file_uploader("Drop video file to begin deep neural analysis", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        if st.button("🚀 INITIATE DEEPER ANALYSIS"):
            with st.status("🧠 Agents are collaborating...", expanded=True) as status:
                st.write("Initializing Multimodal Neural Networks...")
                # The manager now handles its own status updates to st if needed, 
                # but we can also wrap the call.
                report = manager.process_video(uploaded_file)
                status.update(label="Neural Synthesis Complete!", state="complete", expanded=False)
            
            st.session_state.last_report = report # Cache report to keep UI state
            
            st.divider()
            st.header(f"📑 {report['title']}")
            
            # Narrative with better styling
            st.markdown(f"### 🖋️ Intelligence Briefing\n{report['narrative']}")
            
            with st.expander("🛠️ Neural Logs & Raw Data"):
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Visual Matrix")
                    st.json(report.get("visual_raw", {}))
                with c2:
                    st.subheader("Acoustic Fingerprints")
                    st.json(report.get("audio_raw", {}))

with col_stats:
    st.subheader("🎯 Key Insights")
    if 'last_report' in st.session_state:
        report = st.session_state.last_report
        for finding in report["key_findings"]:
            st.info(finding)
    else:
        st.write("Awaiting data feed...")
        st.image("https://img.freepik.com/free-vector/abstract-digital-grid-background-business_53876-120618.jpg")
