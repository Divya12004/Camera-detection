# 🤖 AI-Based Human & Animal Detection

A real-time camera detection system using **YOLOv8, OpenCV, Python, and Flask** to detect humans and animals through a live camera.

## 🚀 Features

- 📷 Live camera detection
- 👤 Human detection
- 🐾 Animal detection
- 🎯 Confidence score
- 🚨 Real-time alert
- 🔊 Alarm notification
- 📸 Detection image saving
- 📝 CSV detection logs
- 🗑️ Delete images older than 5 days
- 🌐 Flask web dashboard

## 🛠️ Technologies

- Python
- YOLOv8
- OpenCV
- Flask
- HTML/CSS/JavaScript
- Linux
- Git & GitHub

## 📁 Project Structure

```text
camera_detection/
├── app.py
├── detector.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── alarm.wav
├── detections/
└── logs/
⚙️ Setup
Clone
git clone YOUR_GITHUB_REPOSITORY_URL
cd camera_detection
Virtual Environment
python3 -m venv venv
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Configure Environment
cp .env.example .env
Run
python app.py

Open:

http://localhost:5000
🔍 Detection
📷 Camera
   ↓
🤖 YOLOv8
   ↓
👤 Human / 🐾 Animal
   ↓
🚨 Alert + 🔊 Alarm
   ↓
📸 Image + 📝 Log
📝 Logs

Detection logs are stored in:

logs/detections.csv

Images are stored in:

detections/

Images older than 5 days are automatically deleted.

🔐 Security

Do not upload:

.env
venv/
detections/
logs/



👩‍💻 Author

Divya Sonawane

B.Tech Computer Science & Engineering


