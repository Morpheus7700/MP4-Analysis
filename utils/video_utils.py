import cv2
import os
from moviepy import VideoFileClip
from PIL import Image
import numpy as np

class VideoUtils:
    @staticmethod
    def extract_audio(video_path, audio_path):
        """Extracts audio from video and saves it to the specified path."""
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    @staticmethod
    def extract_frames(video_path, output_dir, interval=1):
        """Extracts frames from the video at a given interval (in seconds)."""
        frames = []
        try:
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            success, image = vidcap.read()
            count = 0
            
            while success:
                # Calculate time in seconds
                time_s = count / fps
                
                # Capture frame every 'interval' seconds
                if count % int(fps * interval) == 0:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frames.append((time_s, Image.fromarray(image_rgb)))
                
                success, image = vidcap.read()
                count += 1
                
            vidcap.release()
        except Exception as e:
            print(f"Error extracting frames: {e}")
            
        return frames

    @staticmethod
    def get_video_info(video_path):
        """Returns basic metadata about the video."""
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        vidcap.release()
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration
        }
