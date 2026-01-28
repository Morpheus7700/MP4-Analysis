# Video AI Analyst

An autonomous, collaborative AI agent system for analyzing video content.

## Overview
This application uses a multi-agent architecture to process video files:
- **Manager Agent**: Coordinates the workflow.
- **Audio Agent**: Extracts audio and performs speech recognition (Speech-to-Text).
- **Visual Agent**: Extracts frames and performs object/face detection.
- **Report Agent**: Synthesizes findings into a coherent intelligence report.

## Features
- Drag-and-drop interface.
- Automatic Audio/Visual separation.
- Face detection statistics.
- Keyword/Sentiment analysis (basic).
- **Offline Capable**: Visual analysis runs entirely offline. Audio analysis defaults to Google (online) but can be configured for offline use.

## Installation

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: All dependencies should already be installed.*

2.  **Audio Setup (Offline)**:
    - The current setup uses Google's Web Speech API (Online).
    - For offline support, install `pocketsphinx` or `openai-whisper`.
    - To use Whisper, install it (`pip install openai-whisper`) and modify `agents/audio_agent.py`.

## Running the Application

Double-click `run_app.bat` or run the following command:

```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Main frontend.
- `agents/`: Contains the logic for Manager, Audio, Visual, and Report agents.
- `utils/`: Video processing utilities.
