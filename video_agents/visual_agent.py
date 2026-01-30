import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from transformers import pipeline
import mediapipe as mp

class VisualAgent:
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Object Detection (YOLOv8)
        self.yolo_model = YOLO('yolov8n.pt')
        self.yolo_model.to(self.device)
        
        # Determine the best device mapping strategy
        # device_map="auto" is more robust for models that might use meta tensors
        device_map = "auto" if self.device == "cuda" else None
        
        # 2. Image Captioning (BLIP) - Explains what is happening
        self.captioner = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-base",
            device_map=device_map
        )
        
        # 3. Emotion Recognition (ViT) - Understands faces
        try:
            self.emotion_classifier = pipeline(
                "image-classification", 
                model="dima806/facial_emotions_image_detection",
                device_map=device_map
            )
        except Exception as e:
            print(f"Warning: Emotion Recognition model failed to load: {e}")
            self.emotion_classifier = None
        
        # 4. Hand Gesture Recognition (MediaPipe) - Handled with safety for Python 3.13+
        try:
            import mediapipe as mp
            # Use a more resilient way to check for solutions
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
                self.hands_enabled = True
            else:
                print("Warning: MediaPipe 'solutions.hands' not found. Gesture recognition disabled.")
                self.hands_enabled = False
        except (AttributeError, Exception) as e:
            print(f"Warning: MediaPipe Hand solutions initialization failed: {e}")
            self.hands_enabled = False

    def analyze(self, frames):
        """
        Optimized Visual Analysis using Batch Processing and Parallelization.
        """
        if not frames:
            return {"status": "no_data", "insights": ["No frames provided."]}

        insights = []
        object_counts = Counter()
        timeline = [] 
        emotions_detected = Counter()
        gestures_detected = Counter()
        captions_log = []
        
        # Batching parameters
        batch_size = 4
        from utils.video_utils import VideoUtils
        batches = list(VideoUtils.get_batches(frames, batch_size))

        # 1. Batch Object Detection (YOLO) - Very fast
        for batch in batches:
            batch_images = [img for _, img in batch]
            results = self.yolo_model(batch_images, verbose=False)
            
            for i, result in enumerate(results):
                timestamp, _ = batch[i]
                current_objects = []
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    current_objects.append(class_name)
                
                object_counts.update(current_objects)
                # Store placeholder for timeline to be enriched
                timeline.append({"time": timestamp, "objects": current_objects, "caption": "", "emotions": [], "gestures": []})

        # 2. Batch Captioning (BLIP) - Slower, so we sample
        # We process the first frame of every batch for captions (analysis_frequency = batch_size)
        caption_frames = [batch[0][1] for batch in batches]
        try:
            # HuggingFace pipeline supports batching if passed as a list
            # We use a smaller batch for BLIP to avoid OOM
            captions = self.captioner(caption_frames)
            for i, caption_result in enumerate(captions):
                timestamp = batches[i][0][0]
                text = caption_result[0]['generated_text']
                captions_log.append(f"{timestamp:.1f}s: {text}")
                # Assign to the corresponding timeline entry (which is at index i * batch_size)
                timeline[i * batch_size]["caption"] = text
        except Exception as e:
            print(f"Batch Caption Error: {e}")

        # 3. Face & Gestures (Local/CPU intensive, processed per frame but with fast checks)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for i, (timestamp, pil_image) in enumerate(frames):
            img_np = np.array(pil_image)
            
            # --- Emotion Recognition (Only if 'person' detected in this frame) ---
            if 'person' in timeline[i]["objects"] and self.emotion_classifier is not None:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    try:
                        face_roi = pil_image.crop((x, y, x+w, y+h))
                        emotions = self.emotion_classifier(face_roi)
                        top_emotion = emotions[0]['label']
                        timeline[i]["emotions"].append(top_emotion)
                        emotions_detected[top_emotion] += 1
                    except: pass

            # --- Hand Gesture Recognition ---
            if self.hands_enabled:
                results_hands = self.hands.process(img_np)
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        fingers = []
                        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y: fingers.append(1)
                        for fid in [8, 12, 16, 20]:
                            if hand_landmarks.landmark[fid].y < hand_landmarks.landmark[fid - 2].y: fingers.append(1)
                        
                        count = len(fingers)
                        gesture = "Fist" if count == 0 else "Open Palm" if count == 5 else "Pointing" if count == 1 else "Peace Sign" if count == 2 else "Hand Detected"
                        timeline[i]["gestures"].append(gesture)
                        gestures_detected[gesture] += 1

        # 4. Temporal Event Clustering
        temporal_events = []
        if captions_log:
            current_event = {"text": captions_log[0].split(': ')[1], "start": 0.0, "end": 0.0}
            for entry in captions_log[1:]:
                time_s, caption = entry.split(': ')
                time_s = float(time_s.replace('s', ''))
                
                words1 = set(current_event["text"].lower().split())
                words2 = set(caption.lower().split())
                overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
                
                if overlap > 0.5: # Slightly more relaxed overlap for sampled captions
                    current_event["end"] = time_s
                else:
                    temporal_events.append(f"Initially ({current_event['start']:.1f}s), {current_event['text']}.")
                    current_event = {"text": caption, "start": time_s, "end": time_s}
            
            # Use transitions for final narrative
            temporal_events.append(f"Finally ({current_event['start']:.1f}s), {current_event['text']}.")

        # Generate Deep Insights
        gender_mentions = Counter()
        for c in captions_log:
            low_c = c.lower()
            if 'man' in low_c or 'guy' in low_c: gender_mentions['male appearance'] += 1
            if 'woman' in low_c or 'lady' in low_c: gender_mentions['female appearance'] += 1
            if 'child' in low_c or 'boy' in low_c or 'girl' in low_c: gender_mentions['youth appearance'] += 1

        scene_base = captions_log[0].split(': ')[1] if captions_log else 'Analysis unavailable'
        visual_narrative = "Analysis of visual progression: " + " ".join(temporal_events)

        insights.append(f"Primary Scene: {scene_base}.")
        if gender_mentions:
            insights.append(f"Detected: {', '.join(gender_mentions.keys())}.")
        if emotions_detected:
            insights.append(f"Overall Mood: {emotions_detected.most_common(1)[0][0]}.")
        if gestures_detected:
            insights.append(f"Key Actions: {', '.join([k for k, v in gestures_detected.most_common(2)])}.")

        return {
            "agent": "VisualAgent",
            "status": "success",
            "stats": {
                "unique_objects": list(object_counts.keys()),
                "total_emotions": dict(emotions_detected),
                "total_gestures": dict(gestures_detected),
                "visual_narrative": visual_narrative,
                "temporal_events": temporal_events
            },
            "timeline": timeline,
            "captions": captions_log,
            "insights": insights
        }
