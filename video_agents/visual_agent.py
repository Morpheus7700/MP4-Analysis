import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
from transformers import pipeline
import mediapipe as mp

class VisualAgent:
    def __init__(self):
        # 1. Object Detection (YOLOv8)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # 2. Image Captioning (BLIP) - Explains what is happening
        self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        # 3. Emotion Recognition (ViT) - Understands faces
        self.emotion_classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        
        # 4. Hand Gesture Recognition (MediaPipe) - Legacy solutions might be missing on Python 3.13
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            self.hands_enabled = True
        except (AttributeError, Exception) as e:
            print(f"Warning: MediaPipe Hand solutions not available: {e}")
            self.hands_enabled = False

    def analyze(self, frames):
        """
        Comprehensive Visual Analysis: Objects, Actions (Captions), Emotions, Hands.
        """
        insights = []
        object_counts = Counter()
        timeline = [] 
        emotions_detected = Counter()
        gestures_detected = Counter()
        captions_log = []

        if not frames:
            return {"status": "no_data", "insights": ["No frames provided."]}

        # Process Frames
        for timestamp, pil_image in frames:
            frame_data = {"time": timestamp, "objects": [], "caption": "", "emotions": [], "gestures": []}
            
            # --- A. Image Captioning (What is happening?) ---
            # Run only on every 2nd sampled frame to save time, or if list is short
            try:
                # BLIP is relatively fast on CPU
                caption_result = self.captioner(pil_image)
                caption_text = caption_result[0]['generated_text']
                frame_data["caption"] = caption_text
                captions_log.append(f"{timestamp:.1f}s: {caption_text}")
            except Exception as e:
                print(f"Caption Error: {e}")

            # --- B. Object Detection (YOLO) ---
            img_np = np.array(pil_image)
            results = self.yolo_model(pil_image, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    frame_data["objects"].append(class_name)
            
            object_counts.update(frame_data["objects"])

            # --- C. Face Emotion Recognition ---
            # We need to crop faces for the emotion classifier
            # YOLO often detects 'person', but not 'face' specifically.
            # We use OpenCV Haar Cascade for quick face cropping to feed into the Transformer
            # (Or rely on YOLO if we had a face-specific YOLO model, but Haar is fine for cropping here)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                try:
                    face_roi = pil_image.crop((x, y, x+w, y+h))
                    # Emotion Classification
                    emotions = self.emotion_classifier(face_roi)
                    # Get top emotion
                    top_emotion = emotions[0]['label']
                    frame_data["emotions"].append(top_emotion)
                    emotions_detected[top_emotion] += 1
                except:
                    pass

            # --- D. Hand Gesture Recognition ---
            if self.hands_enabled:
                results_hands = self.hands.process(img_np) # MediaPipe expects RGB
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # Simple Logic: Count raised fingers
                        fingers = []
                        # Thumb (check x diff for simplicity or simple geometry)
                        # Tips: 4, 8, 12, 16, 20. PIPs: 2, 6, 10, 14, 18
                        # This is a basic heuristic
                        if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                            fingers.append(1) # Thumb up (approx)
                        
                        # Other 4 fingers (tip y < pip y means raised)
                        for id in [8, 12, 16, 20]:
                            if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                                fingers.append(1)
                        
                        count = len(fingers)
                        if count == 0: gesture = "Fist"
                        elif count == 5: gesture = "Open Palm / Wave"
                        elif count == 1: gesture = "Pointing"
                        elif count == 2: gesture = "Peace Sign / V"
                        else: gesture = "Hand Detected"
                        
                        frame_data["gestures"].append(gesture)
                        gestures_detected[gesture] += 1

            timeline.append(frame_data)

        # Generate Deep Insights
        
        # 1. Narrative Summary
        insights.append(f"Scene Summary: {captions_log[0].split(': ')[1] if captions_log else 'Analysis unavailable'}.")
        
        # 2. Emotions
        if emotions_detected:
            top_emotions = [f"{k} ({v})" for k, v in emotions_detected.most_common(3)]
            insights.append(f"Dominant Emotions detected: {', '.join(top_emotions)}.")
        
        # 3. Gestures
        if gestures_detected:
            top_gestures = [f"{k} ({v})" for k, v in gestures_detected.most_common(3)]
            insights.append(f"Hand signs observed: {', '.join(top_gestures)}.")

        return {
            "agent": "VisualAgent",
            "status": "success",
            "stats": {
                "unique_objects": list(object_counts.keys()),
                "total_emotions": dict(emotions_detected),
                "total_gestures": dict(gestures_detected)
            },
            "timeline": timeline,
            "captions": captions_log,
            "insights": insights
        }
