# youtube_asl_trainer.py - Extract ASL signs from YouTube videos and train model
import yt_dlp
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time
import re

class YouTubeASLTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video URLs with ALL 25 signs per video (150 total signs from 6 videos!)
        # We'll extract samples from entire videos to learn all signs
        self.video_data = [
            {
                'url': 'https://youtu.be/4Ll3OtqAzyw?si=h5rYd5qjJ0iji-2d',
                'signs': ['hello', 'thank_you', 'please', 'sorry', 'yes', 'no', 'good', 'bad', 'help', 
                         'stop', 'go', 'come', 'sit', 'stand', 'walk', 'run', 'eat', 'drink', 'sleep',
                         'wake_up', 'bathroom', 'shower', 'tired', 'hungry', 'thirsty']  # 25 signs
            },
            {
                'url': 'https://youtu.be/gXtdsO_6RQY?si=wtYJf4mvGOGWsCRq', 
                'signs': ['family', 'mother', 'father', 'sister', 'brother', 'baby', 'grandmother', 
                         'grandfather', 'aunt', 'uncle', 'cousin', 'friend', 'people', 'man', 'woman',
                         'boy', 'girl', 'child', 'adult', 'name', 'meet', 'nice', 'you', 'me', 'we']  # 25 signs
            },
            {
                'url': 'https://youtu.be/4QOuEOi_KsY?si=xHsc78Y-LxUh_Jl6',
                'signs': ['home', 'house', 'room', 'bed', 'table', 'chair', 'door', 'window', 'kitchen',
                         'bathroom', 'food', 'water', 'milk', 'bread', 'meat', 'vegetables', 'fruit',
                         'breakfast', 'lunch', 'dinner', 'cook', 'eat', 'drink', 'delicious', 'full']  # 25 signs
            },
            {
                'url': 'https://youtu.be/J5H_HDSipbY?si=aRc4zh6KM3UmPhQI',
                'signs': ['school', 'teacher', 'student', 'learn', 'study', 'book', 'read', 'write',
                         'test', 'homework', 'class', 'red', 'blue', 'green', 'yellow', 'black', 'white',
                         'orange', 'purple', 'pink', 'brown', 'gray', 'color', 'favorite', 'beautiful']  # 25 signs
            },
            {
                'url': 'https://youtu.be/mgLolQA1AZI?si=epqTpwJHCuJBE6Qq',
                'signs': ['time', 'today', 'tomorrow', 'yesterday', 'now', 'later', 'morning', 'afternoon',
                         'evening', 'night', 'day', 'week', 'month', 'year', 'when', 'what', 'where',
                         'who', 'why', 'how', 'question', 'answer', 'ask', 'tell', 'understand']  # 25 signs
            },
            {
                'url': 'https://youtu.be/S4phzFQKzAc?si=-yYGpVXEwC3jjNpl',
                'signs': ['happy', 'sad', 'angry', 'excited', 'scared', 'surprised', 'worried', 'love',
                         'hate', 'like', 'want', 'need', 'have', 'know', 'think', 'feel', 'see', 'hear',
                         'say', 'do', 'make', 'work', 'play', 'fun', 'important']  # 25 signs
            }
        ]
        
    def preprocess_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        lm = []
        for l in hand_landmarks.landmark:
            lm.append([l.x, l.y, l.z])
        lm = np.array(lm)
        
        # Normalize relative to wrist
        wrist = lm[0].copy()
        lm = lm - wrist
        
        # Scale normalization
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm = lm / max_val
        
        return lm.flatten()
    
    def download_video(self, url, output_dir):
        """Download video from YouTube"""
        print(f"Downloading video from: {url}")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Lower quality for faster processing
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"Downloaded: {filename}")
                return filename
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None
    
    def extract_frames_with_hands(self, video_path, max_frames=100):
        """Extract frames that contain hands from video"""
        print(f"Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        frames_with_hands = []
        frame_count = 0
        extracted_count = 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames every 0.5 seconds to avoid too similar frames
        frame_interval = max(1, int(fps * 0.5))
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    # Hand detected, extract landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = self.preprocess_landmarks(hand_landmarks)
                    frames_with_hands.append(landmarks)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        print(f"  Extracted {extracted_count} frames with hands...")
            
            frame_count += 1
        
        cap.release()
        print(f"  Total frames extracted: {len(frames_with_hands)}")
        return frames_with_hands
    
    def process_all_videos(self):
        """Download and process all videos"""
        print("Starting YouTube ASL Training Process...")
        print("="*60)
        
        # Create directories
        video_dir = Path("youtube_videos")
        video_dir.mkdir(exist_ok=True)
        
        all_training_data = []
        
        for i, video_info in enumerate(self.video_data, 1):
            print(f"\nProcessing Video {i}/6...")
            print(f"URL: {video_info['url']}")
            print(f"Expected signs: {', '.join(video_info['signs'])}")
            
            # Download video
            video_path = self.download_video(video_info['url'], video_dir)
            if not video_path:
                continue
            
            # Extract frames with hands (more frames to cover all signs in video)
            landmarks_data = self.extract_frames_with_hands(video_path, max_frames=500)
            
            if landmarks_data:
                # Distribute samples across ALL signs in this video
                num_signs = len(video_info['signs'])
                samples_per_sign = len(landmarks_data) // num_signs
                
                print(f"  Distributing {len(landmarks_data)} samples across {num_signs} signs")
                print(f"  ~{samples_per_sign} samples per sign")
                
                for idx, sign in enumerate(video_info['signs']):
                    start_idx = idx * samples_per_sign
                    end_idx = start_idx + samples_per_sign if idx < num_signs - 1 else len(landmarks_data)
                    
                    sign_samples = landmarks_data[start_idx:end_idx]
                    for landmarks in sign_samples:
                        all_training_data.append((landmarks, sign))
                    
                    print(f"    • {sign}: {len(sign_samples)} samples")
            
            # Clean up video file to save space
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"  Cleaned up video file")
        
        return all_training_data
    
    def save_training_data(self, training_data, filename="youtube_asl_landmarks.csv"):
        """Save training data to CSV"""
        print(f"\nSaving training data to {filename}...")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = []
            for i in range(21):  # 21 hand landmarks
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            header.append('label')
            writer.writerow(header)
            
            # Write data
            for landmarks, label in training_data:
                row = list(landmarks) + [label]
                writer.writerow(row)
        
        print(f"Saved {len(training_data)} samples to {filename}")
    
    def train_model(self, csv_file="youtube_asl_landmarks.csv"):
        """Train model from the extracted data"""
        print(f"\nTraining model from {csv_file}...")
        
        # Load data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples")
        
        # Show class distribution
        print("\nClass distribution:")
        for label, count in df['label'].value_counts().items():
            print(f"  {label}: {count} samples")
        
        # Features and labels
        X = df.drop("label", axis=1).values
        y = df["label"].values
        
        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        # Train model
        print(f"\nTraining Random Forest with {len(X_train)} samples...")
        clf = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        # Save model
        model_path = "youtube_asl_model.joblib"
        joblib.dump({"model": clf, "le": le}, model_path)
        print(f"Model saved to: {model_path}")
        
        return model_path
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("🎬 YouTube ASL Training Pipeline")
        print("="*60)
        print("This will:")
        print("1. Download 6 YouTube ASL videos")
        print("2. Extract frames with hand gestures")
        print("3. Process with MediaPipe to get landmarks")
        print("4. Train an ASL recognition model")
        print("5. Save the trained model")
        print("\nThis may take 10-15 minutes depending on your internet speed...")
        
        input("\nPress Enter to start...")
        
        # Process videos
        training_data = self.process_all_videos()
        
        if not training_data:
            print("❌ No training data extracted!")
            return
        
        # Save data
        self.save_training_data(training_data)
        
        # Train model
        model_path = self.train_model()
        
        print("\n🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Processed {len(training_data)} samples")
        print(f"✅ Model saved to: {model_path}")
        print("\nTo use your new model:")
        print("1. Update real_time.py to use 'youtube_asl_model.joblib'")
        print("2. Run: python real_time.py")

def main():
    trainer = YouTubeASLTrainer()
    trainer.run_full_pipeline()

if __name__ == "__main__":
    main()