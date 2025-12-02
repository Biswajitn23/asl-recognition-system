# 👋 ASL Sign Language Recognition System

Real-time American Sign Language recognition using AI and computer vision. Recognizes **150 ASL signs** trained from YouTube videos.

![ASL Recognition](https://img.shields.io/badge/ASL%20Signs-150-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web app
streamlit run app.py
```

Open `http://localhost:8501` in your browser!

## ✨ Features

- 🎥 **Real-time recognition** through webcam
- 🌐 **Web interface** - Easy to use and share
- 🤖 **150 ASL signs** - Comprehensive vocabulary
- 📚 **YouTube trained** - No manual labeling needed
- 💬 **Sentence builder** - Constructs sentences automatically

## 📊 Recognized Signs

**150 signs across 6 categories:**

- **Greetings** (25): hello, thank_you, please, sorry, yes, no, good, bad, help, stop...
- **Family** (25): family, mother, father, sister, brother, friend, people...
- **Food & Home** (25): home, food, water, eat, drink, breakfast, lunch, dinner...
- **School & Colors** (25): school, teacher, student, red, blue, green, yellow...
- **Time** (25): today, tomorrow, yesterday, when, what, where, who, why...
- **Emotions** (25): happy, sad, love, like, want, need, know, think, feel...

## 🛠️ Tech Stack

- **Python 3.8+**
- **OpenCV** - Video processing
- **MediaPipe** - Hand tracking
- **Scikit-learn** - Machine learning
- **Streamlit** - Web interface
- **yt-dlp** - YouTube video download

## 📁 Project Structure

```
├── app.py                      # Web application
├── youtube_asl_trainer.py      # Train from videos
├── youtube_asl_model.joblib    # Trained model
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 🎓 How It Works

1. **Download** ASL tutorial videos from YouTube
2. **Extract** hand landmarks using MediaPipe
3. **Train** Random Forest model on 3000+ samples
4. **Recognize** signs in real-time through webcam
5. **Build** sentences from recognized gestures

## 🎯 Model Performance

- **Signs**: 150 ASL signs
- **Training samples**: 3,000
- **Accuracy**: 47% (150-class problem)
- **Speed**: Real-time (30+ FPS)

## 🔄 Retrain Model

```bash
# Edit youtube_asl_trainer.py to add video URLs
python youtube_asl_trainer.py
```

## 📱 Usage

1. Launch the web app
2. Allow camera access
3. Show ASL signs to camera
4. Watch real-time recognition
5. Build sentences with gestures!

## 🤝 Contributing

Contributions welcome! Ideas:
- Add more ASL signs
- Improve accuracy
- Enhance UI/UX
- Mobile app version

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- ASL tutorial video creators
- Streamlit for web framework

## 📧 Contact

For questions or suggestions, please open an issue.

---

**Made with ❤️ using Python, OpenCV, and Machine Learning**