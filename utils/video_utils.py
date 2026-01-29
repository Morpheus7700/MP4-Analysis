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
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return False

            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("No audio track found in the video.")
                video.close()
                return False
                
            # Use a slightly more robust ffmpeg parameter if possible
            # But moviepy's write_audiofile is generally okay if audio exists
            video.audio.write_audiofile(audio_path, logger=None, fps=16000) # Standard for Whisper
            video.close() # Close video file handle
            
            # Verify file creation
            import time
            for _ in range(10): # Increased wait time
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    return True
                time.sleep(0.5)
            return False
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    @staticmethod
    def extract_frames(video_path, output_dir, interval=1, max_frames=30):
        """Extracts keyframes using adaptive sampling and scene change detection."""
        frames = []
        try:
            if not os.path.exists(video_path):
                return []
                
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                return []
                
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            
            prev_frame = None
            count = 0
            
            # Intelligent Sampling Parameters
            threshold = 25.0 # Slightly more sensitive
            
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                    
                time_s = count / fps
                
                # Intelligent Scene Change Detection
                # Check every 0.3s for changes (more granular than 0.5s)
                if count % max(1, int(fps * 0.3)) == 0:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    is_keyframe = False
                    if prev_frame is None:
                        is_keyframe = True
                    else:
                        frame_delta = cv2.absdiff(prev_frame, gray)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        change_percent = (np.count_nonzero(thresh) / thresh.size) * 100
                        
                        if change_percent > threshold:
                            is_keyframe = True
                    
                    if is_keyframe or (count % int(fps * 4) == 0): # Force a frame every 4s
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        frames.append((time_s, Image.fromarray(image_rgb)))
                        prev_frame = gray
                
                count += 1
                if len(frames) >= max_frames:
                    break
                    
            vidcap.release()
        except Exception as e:
            print(f"Error extracting frames: {e}")
            
        return frames

    @staticmethod
    def get_batches(items, batch_size):
        """Yield successive n-sized batches from items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    @staticmethod
    def get_video_info(video_path):
        """Returns basic metadata about the video."""
        try:
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check for audio stream presence using moviepy (lightweight check)
            has_audio = False
            try:
                clip = VideoFileClip(video_path)
                has_audio = clip.audio is not None
                clip.close()
            except:
                pass
                
            vidcap.release()
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "has_audio": has_audio
            }
        except:
            return {"fps": 0, "frame_count": 0, "duration": 0, "has_audio": False}
